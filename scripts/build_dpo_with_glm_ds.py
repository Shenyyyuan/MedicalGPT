# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双模型交叉生成 + 交叉评分构建 DPO 数据集
- 生成器: GLM + DeepSeek（外加原始 Qwen 回答作为候选）
- 裁判器: GLM 和 DeepSeek 分别对三个候选打分
- 偏好对构造: 成对比较，分差 > threshold 且 置信度高 的对保留，权重 = 分差 * 置信度
- 输出: dpo_train.jsonl (包含 prompt, chosen, rejected, weight 等)
"""
import os
import sys
import re
import json
import time
import threading
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging
from itertools import combinations

# 强制标准输出 UTF-8
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================= 配置 =======================
HUATUO_DATA_PATH = "/root/autodl-tmp/MedicalGPT/MedicalGPT-v1/HuatuoGPT2_sft_instruct_GPT4_sharegpt.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/MedicalGPT/MedicalGPT-v1/data/cardiovascular"
MAX_SAMPLES = 10          # 测试时可改小，如 10

# GLM 配置
GLM_CONFIG = {
    "api_keys": [k.strip() for k in os.environ.get("GLM_API_KEYS", "").split(",") if k.strip()],
    "model": "glm-5.1",          # 若报错模型不存在，可改为 glm-4-plus
    "api_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
}

# DeepSeek 配置
DEEPSEEK_CONFIG = {
    "api_keys": [k.strip() for k in os.environ.get("DEEPSEEK_API_KEYS", "").split(",") if k.strip()],
    "model": "deepseek-chat",     # DeepSeek 官方模型名
    "api_url": "https://api.deepseek.com/v1/chat/completions",
}

MAX_WORKERS = 2                 # 全局并发数（同时处理的任务数）
RETRY_TIMES = 3
RETRY_BACKOFF = 2
SCORE_DIFF_THRESHOLD = 0.7      # 分差阈值
CONFIDENCE_THRESHOLD = 0.6      # 置信度阈值（0~1，越高要求评委越一致）

# 裁判打分提示词
SCORING_PROMPT_TEMPLATE = """你是一个专业的医学回答质量评估专家。请根据以下标准对给出的回答进行评分（0-10分）：

评分标准：
- 准确性（4分）：回答是否基于正确的医学知识，没有事实错误。
- 完整性（3分）：回答是否完整覆盖了用户问题的关键点，信息量充足。
- 专业性（2分）：是否使用规范的医学术语，表达专业严谨。
- 实用性（1分）：回答是否对用户有实际帮助，建议是否可行。

用户问题：
{question}

待评估的回答：
{answer}

请严格按照以下 JSON 格式输出，只输出 JSON，不要包含其他内容：
{{"score": <分数, 0-10的整数或一位小数>}}
"""

# 生成回答提示词
GENERATION_PROMPT_TEMPLATE = """你是一位经验丰富的医学专家。请针对以下用户问题，提供专业、详细、准确的回答。回答应包含必要的医学解释、可能的病因、建议的检查或处理方式，语言要清晰易懂。

用户问题：{question}

请开始你的回答："""

# ======================= 文本清洗 =======================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[\ud800-\udfff]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,;:!?()【】《》‘’“”\'\"\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ======================= 数据加载 =======================
def load_huatuo_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    logger.info(f"加载数据: {file_path}")
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and len(data_list) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except:
                continue
            prompt = None
            chosen = None
            if "conversations" in obj and isinstance(obj["conversations"], list):
                convs = obj["conversations"]
                for i in range(len(convs)-1):
                    if convs[i].get("from") in ["human","user"] and convs[i+1].get("from") in ["gpt","assistant"]:
                        prompt = convs[i].get("value","")
                        chosen = convs[i+1].get("value","")
                        break
                if prompt is None:
                    human_msgs = [c["value"] for c in convs if c.get("from") in ["human","user"]]
                    gpt_msgs = [c["value"] for c in convs if c.get("from") in ["gpt","assistant"]]
                    if human_msgs and gpt_msgs:
                        prompt = human_msgs[0]
                        chosen = gpt_msgs[-1]
            elif "instruction" in obj and "output" in obj:
                prompt = obj["instruction"]
                chosen = obj["output"]
            elif "prompt" in obj and "chosen" in obj:
                prompt = obj["prompt"]
                chosen = obj["chosen"]
            if not prompt or not chosen:
                continue
            prompt = clean_text(prompt)
            chosen = clean_text(chosen)
            if len(prompt) < 5 or len(chosen) < 300:
                continue
            data_list.append({
                "prompt": prompt,
                "qwen_answer": chosen,   # 原始 SFT Qwen 的回答
                "source": "HuatuoGPT2"
            })
    logger.info(f"加载完成，共 {len(data_list)} 条有效原始数据")
    return data_list

# ======================= API 调用（通用） =======================
def call_api(api_key: str, api_url: str, model: str, prompt: str, temperature: float = 0.7) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    if isinstance(prompt, str):
        prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False
    }
    for attempt in range(1, RETRY_TIMES+1):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.warning(f"API 返回错误状态 {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"请求异常 (attempt {attempt}/{RETRY_TIMES}): {e}")
        if attempt < RETRY_TIMES:
            time.sleep(RETRY_BACKOFF * attempt)
    raise RuntimeError(f"调用 API 失败，重试 {RETRY_TIMES} 次后放弃")

def generate_answer(api_key: str, api_url: str, model: str, question: str) -> str:
    prompt = GENERATION_PROMPT_TEMPLATE.format(question=question)
    return call_api(api_key, api_url, model, prompt, temperature=0.7)

def get_score(api_key: str, api_url: str, model: str, question: str, answer: str) -> float:
    prompt = SCORING_PROMPT_TEMPLATE.format(question=question, answer=answer)
    content = call_api(api_key, api_url, model, prompt, temperature=0.1)
    try:
        content = content.strip()
        if content.startswith('{'):
            data = json.loads(content)
            score = float(data.get("score", 0))
        else:
            match = re.search(r'(\d+(?:\.\d+)?)', content)
            score = float(match.group(1)) if match else 0.0
    except Exception as e:
        logger.warning(f"分数解析失败: {e}, 原始内容: {content[:100]}")
        score = 0.0
    return min(max(score, 0.0), 10.0)

# ======================= 核心：单条处理 =======================
def process_one_item(
    item: Dict,
    glm_key: str,
    deepseek_key: str
) -> List[Dict]:
    """
    返回该样本生成的所有偏好对（列表），每个偏好对格式：
    {
        "prompt": str,
        "chosen": str,
        "rejected": str,
        "weight": float,
        "metadata": {...}
    }
    """
    prompt = item["prompt"]
    qwen_answer = item["qwen_answer"]

    # 1. 生成 GLM 和 DeepSeek 回答
    try:
        glm_answer = generate_answer(
            glm_key, GLM_CONFIG["api_url"], GLM_CONFIG["model"], prompt
        )
        glm_answer = clean_text(glm_answer)
        if len(glm_answer) < 300:
            logger.warning(f"GLM 回答过短，跳过该样本")
            return []
    except Exception as e:
        logger.error(f"GLM 生成失败: {e}, prompt: {prompt[:50]}")
        return []

    try:
        deepseek_answer = generate_answer(
            deepseek_key, DEEPSEEK_CONFIG["api_url"], DEEPSEEK_CONFIG["model"], prompt
        )
        deepseek_answer = clean_text(deepseek_answer)
        if len(deepseek_answer) < 10:
            logger.warning(f"DeepSeek 回答过短，跳过该样本")
            return []
    except Exception as e:
        logger.error(f"DeepSeek 生成失败: {e}, prompt: {prompt[:50]}")
        return []

    # 候选列表: (answer, source)
    candidates = [
        (glm_answer, "GLM"),
        (deepseek_answer, "DeepSeek"),
    ]

    # 2. 两个裁判分别打分
    scores_by_judge = {}  # {judge_name: {source: score}}
    for judge_name, judge_key, judge_config in [
        ("GLM", glm_key, GLM_CONFIG),
        ("DeepSeek", deepseek_key, DEEPSEEK_CONFIG)
    ]:
        score_dict = {}
        for ans, src in candidates:
            try:
                score = get_score(
                    judge_key, judge_config["api_url"], judge_config["model"],
                    prompt, ans
                )
                score_dict[src] = score
            except Exception as e:
                logger.error(f"{judge_name} 打分失败 {src}: {e}")
                return []   # 任何一个打分失败则放弃该样本
        scores_by_judge[judge_name] = score_dict

    # 3. 计算每对候选之间的分差和置信度
    # 分差 = 两个裁判的平均分差值（绝对值）
    # 置信度 = 1 - (两个裁判对该对两个回答的评分差的归一化标准差)，简单起见用 1 - (两个裁判评分的最大差/10)
    #   更鲁棒的方法：对于每个回答，两个裁判的评分一致性高则置信度高；对于一对，取两个回答的一致性最小值。
    # 我们采用：对于候选 a 和 b，定义 confidence = min( 1 - |s_GLM(a)-s_DeepSeek(a)|/10, 1 - |s_GLM(b)-s_DeepSeek(b)|/10 )
    # 即两个裁判对同一回答的评分越接近，置信度越高。

    # 先计算每个候选的两个裁判分数列表
    candidate_scores = {}  # {source: [score_glm, score_deepseek]}
    for src in [c[1] for c in candidates]:
        candidate_scores[src] = [
            scores_by_judge["GLM"][src],
            scores_by_judge["DeepSeek"][src]
        ]

    # 辅助函数：计算置信度（基于两个裁判评分的一致性）
    def calc_confidence(src):
        s1, s2 = candidate_scores[src]
        diff_abs = abs(s1 - s2)
        # diff_abs 范围 0-10，一致性 = 1 - diff_abs/10
        return 1.0 - diff_abs / 10.0

    # 生成所有可能的对
    pairs = []
    for (ans_a, src_a), (ans_b, src_b) in combinations(candidates, 2):
        score_a_glm = scores_by_judge["GLM"][src_a]
        score_a_ds = scores_by_judge["DeepSeek"][src_a]
        score_b_glm = scores_by_judge["GLM"][src_b]
        score_b_ds = scores_by_judge["DeepSeek"][src_b]

        # 平均分
        mean_a = (score_a_glm + score_a_ds) / 2.0
        mean_b = (score_b_glm + score_b_ds) / 2.0
        diff = abs(mean_a - mean_b)

        # 置信度：取两个候选的一致性最小值
        conf_a = calc_confidence(src_a)
        conf_b = calc_confidence(src_b)
        confidence = min(conf_a, conf_b)

        # 筛选条件
        if diff >= SCORE_DIFF_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD:
            # 确定 chosen 和 rejected
            if mean_a > mean_b:
                chosen, rejected = ans_a, ans_b
                chosen_src, rejected_src = src_a, src_b
            else:
                chosen, rejected = ans_b, ans_a
                chosen_src, rejected_src = src_b, src_a
            weight = diff * confidence
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "weight": weight,
                "metadata": {
                    "diff": diff,
                    "confidence": confidence,
                    "scores": {
                        src_a: {"GLM": score_a_glm, "DeepSeek": score_a_ds, "mean": mean_a},
                        src_b: {"GLM": score_b_glm, "DeepSeek": score_b_ds, "mean": mean_b},
                    },
                    "chosen_source": chosen_src,
                    "rejected_source": rejected_src,
                }
            })
    return pairs

# ======================= 多线程处理 =======================
def build_dpo_cross(raw_data: List[Dict], glm_keys: List[str], deepseek_keys: List[str], max_workers: int = 2):
    all_pairs = []
    glm_idx = 0
    deepseek_idx = 0
    lock = threading.Lock()
    progress = {"done": 0, "total": len(raw_data), "lock": threading.Lock()}

    def worker(item):
        nonlocal glm_idx, deepseek_idx
        with lock:
            glm_key = glm_keys[glm_idx % len(glm_keys)]
            glm_idx += 1
            deepseek_key = deepseek_keys[deepseek_idx % len(deepseek_keys)]
            deepseek_idx += 1
        pairs = process_one_item(item, glm_key, deepseek_key)
        with progress["lock"]:
            progress["done"] += 1
            if progress["done"] % 10 == 0:
                logger.info(f"进度: {progress['done']}/{progress['total']}")
        return pairs

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, item) for item in raw_data]
        for future in tqdm(as_completed(futures), total=len(raw_data), desc="构建偏好对"):
            pairs = future.result()
            if pairs:
                all_pairs.extend(pairs)
    return all_pairs

# ======================= 保存数据 =======================
def save_datasets(pairs: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # DPO 数据：可保留 weight 字段（标准 DPO 训练通常不需要 weight，但可以存储备用）
    dpo_path = os.path.join(output_dir, "dpo_train.jsonl")
    with open(dpo_path, "w", encoding="utf-8") as f:
        for p in pairs:
            # 输出标准格式 + weight（可选）
            record = {
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
                "weight": p["weight"]   # 可被自定义 loss 使用
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"已保存 DPO 数据: {len(pairs)} 条 -> {dpo_path}")

    # 同时保存完整的元数据版本（用于分析）
    meta_path = os.path.join(output_dir, "dpo_metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info(f"已保存完整元数据: {meta_path}")

    # 生成 SFT 格式（使用每个 pair 的 chosen 作为 output，但可能有重复 prompt，取最好的 chosen？这里简单处理：每个 pair 生成一条 SFT）
    sft_path = os.path.join(output_dir, "sft_train.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"instruction": p["prompt"], "output": p["chosen"]}, ensure_ascii=False) + "\n")
    logger.info(f"已保存 SFT 数据: {len(pairs)} 条 -> {sft_path}")

    # ORPO 数据格式与 DPO 相同
    orpo_path = os.path.join(output_dir, "orpo_train.jsonl")
    with open(orpo_path, "w", encoding="utf-8") as f:
        for p in pairs:
            record = {
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"已保存 ORPO 数据: {len(pairs)} 条 -> {orpo_path}")

    # 可选：预训练文本
    pretrain_path = os.path.join(output_dir, "pretrain.txt")
    with open(pretrain_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(f"用户问题：{p['prompt']}\n医生回答：{p['chosen']}\n\n")
    logger.info(f"已保存预训练文本: {pretrain_path}")

    # 统计信息
    stats = {
        "total_pairs": len(pairs),
        "method": "CrossGeneration_CrossJudge_PairwiseWeighted",
        "glm_model": GLM_CONFIG["model"],
        "deepseek_model": DEEPSEEK_CONFIG["model"],
        "score_diff_threshold": SCORE_DIFF_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }
    with open(os.path.join(output_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

# ======================= 主函数 =======================
def main():
    logger.info("="*60)
    logger.info("开始双模型交叉生成+交叉评分构建 DPO 数据集")
    logger.info("="*60)

    raw_data = load_huatuo_data(HUATUO_DATA_PATH, max_samples=MAX_SAMPLES)
    if not raw_data:
        return

    glm_keys = GLM_CONFIG["api_keys"]
    deepseek_keys = DEEPSEEK_CONFIG["api_keys"]
    if not glm_keys or not deepseek_keys:
        logger.error("请配置 GLM 和 DeepSeek 的 API Keys")
        return

    logger.info(f"原始样本数: {len(raw_data)}")
    logger.info(f"GLM keys: {len(glm_keys)}, DeepSeek keys: {len(deepseek_keys)}")
    logger.info(f"并发数: {MAX_WORKERS}, 分差阈值: {SCORE_DIFF_THRESHOLD}, 置信度阈值: {CONFIDENCE_THRESHOLD}")

    pairs = build_dpo_cross(raw_data, glm_keys, deepseek_keys, max_workers=MAX_WORKERS)
    if pairs:
        save_datasets(pairs, OUTPUT_DIR)
        logger.info(f"完成！共生成 {len(pairs)} 个偏好对。")
    else:
        logger.error("未生成任何偏好对，请检查 API 配置或降低阈值。")

if __name__ == "__main__":
    main()