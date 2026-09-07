# ============================================================
# Step 1: Train Reward Model (基于原始基座模型，用DPO同源数据)
# 在 DPO 训练后，用偏好数据训练一个 Reward Model
# ============================================================

cd /root/autodl-tmp/MedicalGPT/MedicalGPT-v1

# RM 必须基于原始基座模型（与SFT/DPO同源），而不是基于DPO模型
# 数据格式与DPO相同：{conversations: [...], chosen: "...", rejected: "..."}

CUDA_VISIBLE_DEVICES=0 python training/reward_modeling.py \
    --model_name_or_path Qwen/Qwen3.5-4B \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 20 \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --max_source_length 2048 \
    --max_target_length 512 \
    --output_dir ./output/reward_model_after_dpo_v3 \
    --overwrite_output_dir \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True
