# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import torch


def main():
    project_root = "/root/autodl-tmp/MedicalGPT/MedicalGPT-v1"
    base_model = f"{project_root}/models/Qwen3.5-4B"
    dpo_merged = f"{project_root}/outputs/dpo_v3_merged_full"
    reward_model_dir = f"{project_root}/output/reward_model_after_dpo_v3"
    ppo_data_dir = f"{project_root}/data/reward/rm_data"
    output_dir = f"{project_root}/output/ppo_from_dpo_v3"

    os.chdir(project_root)

    # Environment settings for server
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # ?? datasets ??
    cache_dir = os.path.join(project_root, "cache", "datasets_cache")
    os.makedirs(cache_dir, exist_ok=True)  # Preserve datasets cache across runs.

    nproc = torch.cuda.device_count()
    master_port = "29507"

    cmd = [
        "torchrun", "--nproc_per_node", str(nproc), "--master_port", master_port,
        "training/ppo_training.py",

        # Model paths
        "--model_name_or_path", base_model,
        "--sft_model_path", dpo_merged,
        "--reward_model_path", reward_model_dir,

        # Data
        "--train_file_dir", ppo_data_dir,
        "--validation_file_dir", os.environ["MEDICAL_VALIDATION_DIR"],
        "--max_source_length", "512",

        # Training
        "--do_train",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--generation_batch_size", "6",
        "--gradient_checkpointing", "True",
        "--bf16", "True",
        "--max_steps", "100",
        "--logging_steps", "10",
        "--save_steps", "50",
        "--remove_unused_columns", "False",

        # LoRA
        "--use_peft", "True",
        "--lora_r", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",

        # Output
        "--cache_dir", cache_dir,
        "--output_dir", output_dir,
    ]

    print("=" * 60)
    print("Step 2: PPO Training (from DPO model, torchrun)")
    print(f"Base model (for RM): {base_model}")
    print(f"Policy start (DPO):  {dpo_merged}")
    print(f"Reward model LoRA:   {reward_model_dir}")
    print(f"Data:                {ppo_data_dir}")
    print(f"Output:              {output_dir}")
    print("=" * 60)
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
