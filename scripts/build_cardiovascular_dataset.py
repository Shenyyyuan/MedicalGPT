#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
心血管疾病微调数据集构建脚本（整合版）
数据源：HuatuoGPT2_sft_instruct_GPT4_sharegpt.jsonl
功能：数据加载、标准化、去重、长度过滤、格式转换（SFT/DPO/ORPO）
注意：已移除心血管关键词筛选，保留所有原始医疗问答对
"""
import chardet
import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging
from sklearn.utils import shuffle

# 配置日志
logging.basicConfig (level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger (__name__)

# ======================= 配置参数 =======================
# 数据源路径
HUATUO_DATA_PATH = "/root/autodl-tmp/MedicalGPT/MedicalGPT-v1/HuatuoGPT2_sft_instruct_GPT4_sharegpt.jsonl"
# 输出目录
OUTPUT_DIR = "/root/autodl-tmp/MedicalGPT/MedicalGPT-v1/data/cardiovascular"

USE_HUATUO = True  # 使用 HuatuoGPT2 数据源
SAMPLE_SIZE = 5000  # 最终数据集目标大小（5,000 条）


# ======================= 辅助函数 =======================
def basic_text_clean(text: str) -> str:
    """文本基础清洗：保留中文、字母、数字、空格以及常见标点（中英文）"""
    if not isinstance (text, str):
        return ""
    # 去除 HTML 标签
    text = re.sub (r'<[^>]+>', '', text)
    # 只移除控制字符（ASCII 0-31 除了制表符、换行符等），保留所有可见字符包括标点
    text = re.sub (r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 多余空白处理（多个空格合并为一个，首尾去空格）
    text = re.sub (r'\s+', ' ', text).strip ()
    return text


def normalize_medical_terms(text: str) -> str:
    """医学术语标准化（扩展至血液病）"""
    term_mapping = {
        # 心血管原有映射（保留）
        "高血压病": "高血压", "血压高": "高血压",
        "冠心病": "冠状动脉粥样硬化性心脏病",
        "心肌梗死": "急性心肌梗死", "心梗": "急性心肌梗死",
        "心力衰竭": "充血性心力衰竭", "心衰": "充血性心力衰竭",
        "心律失常": "心律不齐", "房颤": "心房颤动",
        "心绞痛": "稳定性心绞痛",
        "胸闷": "胸闷气短", "心悸": "心慌心悸",
        "心电图": "十二导联心电图", "ECG": "心电图", "EKG": "心电图",
        "冠脉造影": "冠状动脉造影",
        "阿司匹林": "阿司匹林肠溶片", "降压药": "抗高血压药物",
        "他汀": "他汀类药物",

        # 新增：血液/淋巴系统（区分白血病 vs 淋巴瘤）
        "淋巴白血病": "淋巴细胞白血病",
        "急性淋巴白血病": "急性淋巴细胞白血病（ALL）",
        "慢性淋巴白血病": "慢性淋巴细胞白血病（CLL）",
        "淋巴瘤": "淋巴瘤（注意：不同于白血病）",
        "霍奇金淋巴瘤": "霍奇金淋巴瘤",
        "非霍奇金淋巴瘤": "非霍奇金淋巴瘤",

        # 常见混淆词纠正（可选，谨慎使用）
        "淋巴癌": "淋巴瘤",  # 公众俗称，纠正为医学名称
    }
    result = text
    # 按长度降序替换，避免短词覆盖长词
    for term, normalized in sorted (term_mapping.items (), key=lambda x: len (x[0]), reverse=True):
        pattern = re.compile (re.escape (term), re.IGNORECASE)
        result = pattern.sub (normalized, result)
    return result


def filter_valid_example(prompt: str, chosen: str, rejected: str, min_length: int = 300) -> bool:
    """验证单条数据的有效性 - 提高min_length阈值"""
    if not isinstance (prompt, str) or len (prompt.strip ()) < 5:
        return False
    if not isinstance (chosen, str) or len (chosen.strip ()) < min_length:
        return False
    if not isinstance (rejected, str) or len (rejected.strip ()) < min_length:
        return False
    return True


def save_jsonl(data: List[Dict], output_path: str):
    """保存为JSONL格式"""
    os.makedirs (os.path.dirname (output_path), exist_ok=True)
    with open (output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write (json.dumps (item, ensure_ascii=False) + '\n')
    logger.info (f"已保存 {len (data)} 条数据到 {output_path}")


def read_csv_with_fallback(csv_path: str) -> pd.DataFrame:
    """保留备用（不再使用）"""
    with open (csv_path, 'rb') as f:
        raw_sample = f.read (65536)
    detected = chardet.detect (raw_sample).get ('encoding')
    candidate_encodings = ['gb18030', 'gbk', detected, 'utf-8-sig', 'utf-8']
    tried = []
    for encoding in candidate_encodings:
        if not encoding:
            continue
        encoding = encoding.lower ()
        if encoding in tried:
            continue
        tried.append (encoding)
        try:
            return pd.read_csv (csv_path, encoding=encoding, engine='python')
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError (
        "unknown",
        b"",
        0,
        1,
        f"无法用常见编码读取 CSV，已尝试: {', '.join (tried)}"
    )


# ======================= 1. 加载 HuatuoGPT2 数据 =======================
def load_huatuo_data(file_path: str) -> List[Dict]:
    """
    加载 HuatuoGPT2 数据（ShareGPT 格式），提取 prompt 和 chosen，并生成 rejected。
    支持的字段格式：
        - conversations: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
        - instruction + output
        - prompt + chosen
    """
    if not os.path.exists (file_path):
        logger.error (f"数据文件不存在: {file_path}")
        return []

    logger.info (f"正在加载 HuatuoGPT2 数据: {file_path}")
    data_list = []
    with open (file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate (tqdm (f, desc="读取 HuatuoGPT2 数据"), 1):
            line = line.strip ()
            if not line:
                continue
            try:
                obj = json.loads (line)
            except json.JSONDecodeError:
                logger.warning (f"第 {line_num} 行 JSON 解析失败，跳过")
                continue

            prompt = None
            chosen = None

            # 1) 尝试 conversations 格式
            if "conversations" in obj and isinstance (obj["conversations"], list):
                convs = obj["conversations"]
                # 查找第一组 human->gpt 对话
                for i in range (len (convs) - 1):
                    if convs[i].get ("from") in ["human", "user"] and convs[i + 1].get ("from") in ["gpt", "assistant"]:
                        prompt = convs[i].get ("value", "")
                        chosen = convs[i + 1].get ("value", "")
                        break
                # 如果没找到，尝试只取第一个 human 和最后一个 gpt
                if prompt is None:
                    human_msgs = [c["value"] for c in convs if c.get ("from") in ["human", "user"]]
                    gpt_msgs = [c["value"] for c in convs if c.get ("from") in ["gpt", "assistant"]]
                    if human_msgs and gpt_msgs:
                        prompt = human_msgs[0]
                        chosen = gpt_msgs[-1]

            # 2) 尝试 instruction/output 格式
            elif "instruction" in obj and "output" in obj:
                prompt = obj["instruction"]
                chosen = obj["output"]

            # 3) 尝试 prompt/chosen 格式
            elif "prompt" in obj and "chosen" in obj:
                prompt = obj["prompt"]
                chosen = obj["chosen"]

            if not prompt or not chosen:
                logger.debug (f"第 {line_num} 行缺少有效问答对，跳过")
                continue

            # 基础清洗
            prompt = basic_text_clean (prompt)
            chosen = basic_text_clean (chosen)
            if len (prompt) < 5 or len (chosen) < 10:
                continue

            # 生成 rejected（简化版，与原逻辑一致）
            if len (chosen) > 50:
                rejected = chosen[:len (chosen) // 3] + "..." if len (chosen) // 3 > 20 else chosen[:30] + "..."
            else:
                rejected = chosen[:min (30, len (chosen))] + "（回答不完整）"

            data_list.append ({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "HuatuoGPT2",
                "raw": obj  # 保留原始对象，便于调试
            })

    logger.info (f"HuatuoGPT2 数据加载完成，共 {len (data_list)} 条原始记录")
    return data_list


# ======================= 2. 通用数据清洗函数 =======================
# 注意：已移除心血管关键词筛选函数 filter_cardiovascular

def deduplicate_data(data_list: List[Dict]) -> List[Dict]:
    """基于 prompt 和 chosen 进行去重"""
    seen = set ()
    deduped = []
    for item in data_list:
        key = f"{item['prompt'][:100]}::{item['chosen'][:100]}"
        if key not in seen:
            seen.add (key)
            deduped.append (item)
    logger.info (f"去重后保留 {len (deduped)} 条数据")
    return deduped


def filter_by_length(data_list: List[Dict], min_prompt_len: int = 5,
                     max_prompt_len: int = 2000, min_response_len: int = 10,
                     max_response_len: int = 2048) -> List[Dict]:
    """长度过滤"""
    filtered = []
    for item in data_list:
        prompt_len = len (item.get ('prompt', ''))
        chosen_len = len (item.get ('chosen', ''))
        if (min_prompt_len <= prompt_len <= max_prompt_len and
                chosen_len >= min_response_len and
                chosen_len <= max_response_len):
            filtered.append (item)
    logger.info (f"长度过滤后保留 {len (filtered)} 条数据")
    return filtered


# ======================= 3. 格式转换 =======================
def convert_to_sft_format(data_list: List[Dict]) -> List[Dict]:
    """转换为SFT格式: instruction, output"""
    sft_data = []
    for item in data_list:
        sft_data.append ({
            "instruction": item["prompt"],
            "output": item["chosen"]
        })
    return sft_data


def prepare_grpo_reward_config() -> Dict:
    """准备GRPO奖励函数配置"""
    return {
        "json_format_bonus": 0.2,
        "keyword_bonus": 0.3,
        "grounding_bonus": 0.4,
        "length_penalty": 0.1,
        "forbidden_words": ["不确定", "可能", "大概", "也许", "或许", "好像"]
    }


# ======================= 4. 主流程 =======================
def build_full_pipeline(target_size: int = 20000, use_huatuo: bool = True):
    """完整的数据处理流水线（只使用 HuatuoGPT2 数据源，无心血管筛选）"""
    logger.info ("=" * 50)
    logger.info ("开始构建医疗微调数据集（HuatuoGPT2 数据源，无关键词筛选）")
    logger.info ("=" * 50)

    all_data = []

    # 加载 HuatuoGPT2 数据
    if use_huatuo:
        raw_huatuo = load_huatuo_data (HUATUO_DATA_PATH)
        if raw_huatuo:
            # 不再进行关键词筛选，直接使用全部数据
            all_data.extend (raw_huatuo)
            logger.info (f"HuatuoGPT2 数据（全部）: {len (raw_huatuo)} 条")
        else:
            logger.warning ("未加载到任何 HuatuoGPT2 数据")
    else:
        logger.warning ("未启用任何数据源，无法构建数据集")
        return None

    if not all_data:
        logger.error ("没有加载到任何数据！")
        return None

    # 3. 统一清洗流程
    logger.info ("开始统一数据清洗...")
    # 长度过滤
    all_data = filter_by_length (all_data, min_prompt_len=5, max_prompt_len=2000, min_response_len=10)
    # 去重
    all_data = deduplicate_data (all_data)
    # 术语标准化（可选，保留但不会改变语义）
    for item in tqdm (all_data, desc="医学术语标准化"):
        item["prompt"] = normalize_medical_terms (item["prompt"])
        item["chosen"] = normalize_medical_terms (item["chosen"])
        item["rejected"] = normalize_medical_terms (item["rejected"])

    # 5. 采样到目标数量
    if len (all_data) > target_size:
        all_data = shuffle (all_data, random_state=42)[:target_size]
        logger.info (f"采样至目标数量 {target_size} 条")
    else:
        logger.info (f"数据总量 {len (all_data)} 条，不足目标数量 {target_size}，将使用全部数据")

    # 6. 生成各阶段训练数据
    sft_data = convert_to_sft_format (all_data)
    dpo_data = all_data  # 已经符合 DPO 格式
    orpo_data = all_data

    # 7. 预训练文本数据（纯文本格式）
    pretrain_texts = []
    for item in all_data:
        text = f"用户问题：{item['prompt']}\n医生回答：{item['chosen']}\n"
        pretrain_texts.append (text)

    # 8. GRPO 数据
    grpo_data = sft_data
    reward_config = prepare_grpo_reward_config ()

    result = {
        "total_data": len (all_data),
        "huatuo_count": len ([d for d in all_data if d.get ('source') == 'HuatuoGPT2']),
        "pretrain_data": pretrain_texts,
        "sft_data": sft_data,
        "dpo_data": dpo_data,
        "orpo_data": orpo_data,
        "grpo_data": grpo_data,
        "reward_config": reward_config,
        "raw_dpo_data": dpo_data,
        "stats": {
            "total_samples": len (all_data),
            "avg_prompt_length": np.mean ([len (d["prompt"]) for d in all_data]),
            "avg_chosen_length": np.mean ([len (d["chosen"]) for d in all_data]),
            "avg_rejected_length": np.mean ([len (d["rejected"]) for d in all_data])
        }
    }

    # 9. 打印统计信息
    logger.info ("\n" + "=" * 50)
    logger.info ("数据集构建完成！统计信息：")
    logger.info (f"  总数据量: {result['total_data']} 条")
    logger.info (f"  HuatuoGPT2 来源: {result['huatuo_count']} 条")
    logger.info (f"  平均 Prompt 长度: {result['stats']['avg_prompt_length']:.1f} 字符")
    logger.info (f"  平均 Chosen 长度: {result['stats']['avg_chosen_length']:.1f} 字符")
    logger.info (f"  平均 Rejected 长度: {result['stats']['avg_rejected_length']:.1f} 字符")
    logger.info ("=" * 50)

    return result


def save_all_datasets(results: Dict[str, any], output_dir: str = "./data/cardiovascular"):
    """保存所有数据集到指定目录"""
    os.makedirs (output_dir, exist_ok=True)

    # 1. 保存预训练数据
    with open (os.path.join (output_dir, "pretrain.txt"), "w", encoding="utf-8") as f:
        for text in results["pretrain_data"]:
            f.write (text + "\n")

    # 2. 保存 SFT 数据
    save_jsonl (results["sft_data"], os.path.join (output_dir, "sft_train.jsonl"))

    # 3. 保存 DPO 数据
    save_jsonl (results["dpo_data"], os.path.join (output_dir, "dpo_train.jsonl"))

    # 4. 保存 ORPO 数据
    save_jsonl (results["orpo_data"], os.path.join (output_dir, "orpo_train.jsonl"))

    # 5. 保存 GRPO 数据和奖励配置
    with open (os.path.join (output_dir, "grpo_data.json"), "w", encoding="utf-8") as f:
        json.dump (results["grpo_data"], f, ensure_ascii=False, indent=2)

    with open (os.path.join (output_dir, "reward_config.json"), "w", encoding="utf-8") as f:
        json.dump (results["reward_config"], f, ensure_ascii=False, indent=2)

    # 6. 保存统计信息
    stats = {
        "total_samples": results["total_data"],
        "huatuo_count": results["huatuo_count"],
        "avg_prompt_length": results["stats"]["avg_prompt_length"],
        "avg_chosen_length": results["stats"]["avg_chosen_length"],
        "avg_rejected_length": results["stats"]["avg_rejected_length"]
    }
    with open (os.path.join (output_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump (stats, f, ensure_ascii=False, indent=2)

    logger.info (f"所有数据集已保存到 {output_dir}")


def verify_dataset_format(output_dir: str = "./data/cardiovascular"):
    """验证生成的数据格式是否正确"""
    logger.info ("正在验证数据集格式...")

    # 验证 SFT 数据
    sft_path = os.path.join (output_dir, "sft_train.jsonl")
    if os.path.exists (sft_path):
        with open (sft_path, "r", encoding="utf-8") as f:
            first_line = f.readline ()
            sample = json.loads (first_line)
            assert "instruction" in sample and "output" in sample, "SFT 数据格式错误"
        logger.info ("✓ SFT 数据格式验证通过")

    # 验证 DPO 数据
    dpo_path = os.path.join (output_dir, "dpo_train.jsonl")
    if os.path.exists (dpo_path):
        with open (dpo_path, "r", encoding="utf-8") as f:
            first_line = f.readline ()
            sample = json.loads (first_line)
            assert all (k in sample for k in ["prompt", "chosen", "rejected"]), "DPO 数据格式错误"
        logger.info ("✓ DPO 数据格式验证通过")

    logger.info ("所有数据集格式验证完成")


# ======================= 执行入口 =======================
if __name__ == "__main__":
    # 构建完整的数据集（只使用 HuatuoGPT2 数据，目标 20000 条，无关键词筛选）
    results = build_full_pipeline (target_size=SAMPLE_SIZE, use_huatuo=USE_HUATUO)

    if results:
        # 保存所有数据集
        save_all_datasets (results, OUTPUT_DIR)
        # 验证数据格式
        verify_dataset_format (OUTPUT_DIR)

        # 打印示例数据
        logger.info ("\n示例 SFT 数据：")
        logger.info (json.dumps (results["sft_data"][0], ensure_ascii=False, indent=2))

        logger.info ("\n示例 DPO 数据：")
        logger.info (json.dumps (results["dpo_data"][0], ensure_ascii=False, indent=2))