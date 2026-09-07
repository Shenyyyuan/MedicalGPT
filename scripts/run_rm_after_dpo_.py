# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import torch


def main():
    project_root = "/root/autodl-tmp/MedicalGPT/MedicalGPT-v1"
    os.chdir(project_root)

    # 环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用 2 张卡

    # 清理 datasets 缓存，避免旧 schema 污染
    cache_dir = os.path.join(project_root, "cache", "datasets_cache")
    subprocess.run(["rm", "-rf", cache_dir])

    nproc = torch.cuda.device_count()
    master_port = "29507"

    cmd = [
        "torchrun", "--nproc_per_node", str(nproc), "--master_port", master_port,
        "training/reward_modeling.py",

        # Model
        "--model_name_or_path", f"{project_root}/models/Qwen3.5-4B",
        "--torch_dtype", "bfloat16",
        "--trust_remote_code", "True",

        # Data
        "--train_file_dir", "./data/reward",
        "--validation_file_dir", "./data/reward",
        "--max_source_length", "2048",
        "--max_target_length", "512",

        # Training
        "--do_train",
        "--per_device_train_batch_size", "4",
        "--gradient_accumulation_steps", "4",
        "--num_train_epochs", "1",
        "--learning_rate", "2e-5",
        "--warmup_steps", "20",
        "--logging_steps", "10",
        "--save_steps", "100",
        "--save_total_limit", "3",
        "--seed", "42",
        "--bf16",
        "--gradient_checkpointing", "True",
        "--remove_unused_columns", "False",
        "--report_to", "tensorboard",

        # DDP / 多卡
        "--ddp_find_unused_parameters", "False",

        # LoRA
        "--use_peft", "True",
        "--target_modules", "all",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",

        # Output
        "--deepspeed", "./scripts/ds_config_rm.json",
        "--cache_dir", cache_dir,
        "--output_dir", "./output/reward_model_after_dpo_v3",
    ]

    print("=" * 60)
    print("Step 1: Training Reward Model (torchrun)")
    print(f"Base model: {cmd[cmd.index('--model_name_or_path') + 1]}")
    print(f"Data: {cmd[cmd.index('--train_file_dir') + 1]}")
    print(f"Output: {cmd[cmd.index('--output_dir') + 1]}")
    print("=" * 60)
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
