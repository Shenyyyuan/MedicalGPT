# ============================================================
# Step 2: PPO Training (从 DPO 合并模型开始，用 RM 进一步优化)
# DPO 模型作为 policy 起点，RM 提供奖励信号
# ============================================================

cd /root/autodl-tmp/MedicalGPT/MedicalGPT-v1
unset OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python training/ppo_training.py \
    --model_name_or_path Qwen/Qwen3.5-4B \
    --sft_model_path /root/autodl-tmp/MedicalGPT/MedicalGPT-v1/outputs/dpo_v3_merged_full \
    --reward_model_path /root/autodl-tmp/MedicalGPT/MedicalGPT-v1/output/reward_model_after_dpo_v3 \
    --train_file_dir /root/autodl-tmp/MedicalGPT/MedicalGPT-v1/data/reward/rm_data \
    --validation_file_dir /root/autodl-tmp/MedicalGPT/MedicalGPT-v1/data/reward/rm_data \
    --output_dir /root/autodl-tmp/MedicalGPT/MedicalGPT-v1/output/ppo_from_dpo_v3 \
    --do_train \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --generation_batch_size 2 \
    --gradient_checkpointing True \
    --bf16 True \
    --use_peft True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_source_length 512 \
    --remove_unused_columns False \
    --max_steps 100 \
    --logging_steps 10 \
    --save_steps 50
