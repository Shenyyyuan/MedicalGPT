# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train a model from SFT using RLOO (REINFORCE Leave-One-Out, PPO alternative)
"""

import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
import torch
from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
)
from peft import PeftModel
from trl import (
    RLOOConfig,
    RLOOTrainer,
    ModelConfig,
    get_peft_config,
)

sys.path.append (os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
from training.template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class RLOOArguments:
    sft_model_path: Optional[str] = field (default=None, metadata={"help": "Path to the SFT model."})
    policy_adapter_path: Optional[str] = field (default=None, metadata={"help": "Path to the policy LoRA adapter, e.g. DPO adapter."})
    reward_model_path: Optional[str] = field (default=None, metadata={"help": "Path to the reward model LoRA."})
    dataset_name: Optional[str] = field (default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field (default=None, metadata={"help": "Dataset configuration name."})
    dataset_train_split: str = field (default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field (default="test", metadata={"help": "Dataset split to use for evaluation."})
    train_file_dir: Optional[str] = field (default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field (default=None, metadata={"help": "The evaluation jsonl file folder."})
    template_name: Optional[str] = field (default=None, metadata={"help": "The prompt template name."})
    max_source_length: Optional[int] = field (default=1024, metadata={"help": "Max length of prompt input text"})

def sync_model_special_token_ids(model, tokenizer):
    token_ids = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    for module in model.modules():
        for obj in (getattr(module, "config", None), getattr(module, "generation_config", None)):
            if obj is None:
                continue
            for nested in (obj, getattr(obj, "text_config", None)):
                if nested is None:
                    continue
                for name, value in token_ids.items():
                    if value is not None and hasattr(nested, name):
                        setattr(nested, name, value)

def main():
    parser = HfArgumentParser ((RLOOArguments, RLOOConfig, ModelConfig))
    args, training_args, model_args = parser.parse_args_into_dataclasses (return_remaining_strings=True)[:3]

    local_rank = int (os.environ.get ("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0
    world_size = int (os.environ.get ("WORLD_SIZE", "1"))
    ddp = world_size != 1
    num_gpus = torch.cuda.device_count ()

    if is_main_process:
        logger.info (f"Parse args: {args}")
        logger.info (f"Training args: {training_args}")
        logger.info (f"Model args: {model_args}")
        logger.info (f"DDP: {ddp}, num_gpus: {num_gpus}")

    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr (torch, model_args.dtype)
    )

    # Load tokenizer
    sft_model_path = args.sft_model_path or model_args.model_name_or_path
    tokenizer_path = model_args.model_name_or_path if args.policy_adapter_path else sft_model_path
    tokenizer = AutoTokenizer.from_pretrained (tokenizer_path, trust_remote_code=model_args.trust_remote_code)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.sep_token
        tokenizer.add_special_tokens ({"eos_token": tokenizer.eos_token})
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens ({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # Configure the model consistently with its training checkpoint.
    reward_base_path = model_args.model_name_or_path  # Configure the model consistently with its training checkpoint.
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig (load_in_4bit=True)  # 使用 4bit 量化，可节约显存

    # Load reward model (LoRA + 4bit)
    logger.info (f"Loading reward model from LoRA path: {args.reward_model_path}")
    reward_model = AutoModelForSequenceClassification.from_pretrained (
        reward_base_path,
        num_labels=1,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
        device_map=None,
    )
    reward_model = PeftModel.from_pretrained (reward_model, args.reward_model_path)
    device = training_args.device if hasattr (training_args, 'device') else torch.device ("cuda")
    reward_model = reward_model.to (device)
    reward_model.eval ()
    for p in reward_model.parameters ():
        p.requires_grad = False

    # Configure the model consistently with its training checkpoint.
    sync_model_special_token_ids(reward_model, tokenizer)

    # Load policy model
    policy_base_path = model_args.model_name_or_path if args.policy_adapter_path else sft_model_path
    logger.info (f"Loading policy base model from: {policy_base_path}")
    policy = AutoModelForCausalLM.from_pretrained (
        policy_base_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
    )
    if args.policy_adapter_path:
        logger.info (f"Loading policy adapter from: {args.policy_adapter_path}")
        policy = PeftModel.from_pretrained (policy, args.policy_adapter_path, is_trainable=False)
        policy = policy.merge_and_unload ()
    policy = policy.to (device)
    # Configure the model consistently with its training checkpoint.
    sync_model_special_token_ids(policy, tokenizer)
    policy.gradient_checkpointing_enable ()
    policy.enable_input_require_grads ()
    policy.config.use_cache = False

    # Load PEFT config for policy (LoRA PPO)
    peft_config = get_peft_config (model_args)

    # Load dataset
    prompt_template = get_conv_template (args.template_name) if args.template_name else None
    if args.dataset_name:
        dataset = load_dataset (args.dataset_name, args.dataset_config, split=args.dataset_train_split)
        eval_samples = 100
        train_dataset = dataset.select (range (len (dataset) - eval_samples))
        eval_dataset = dataset.select (range (len (dataset) - eval_samples, len (dataset)))
    else:
        data_files = {}
        if args.train_file_dir and os.path.exists (args.train_file_dir):
            data_files["train"] = glob (f'{args.train_file_dir}/**/*.jsonl', recursive=True)
        if args.validation_file_dir and os.path.exists (args.validation_file_dir):
            data_files["validation"] = glob (f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
        dataset = load_dataset ('json', data_files=data_files)
        train_dataset = dataset["train"]
        if "validation" not in dataset:
            dataset = dataset["train"].train_test_split (test_size=0.05, seed=42)
            train_dataset = dataset["train"]
            val_dataset = dataset["test"]
        else:
            train_dataset = dataset["train"]
            val_dataset = dataset["validation"]
        eval_dataset = val_dataset.select (range (min (100, len (val_dataset))))

    # Preprocessing
    max_source_length = args.max_source_length

    def preprocess_function(examples):
        new_examples = {"prompt": []}
        for conv in examples["conversations"]:
            prompt = None
            for turn in conv:
                if turn.get ("from") == "human":
                    prompt = turn.get ("value")
                    break
            if prompt:
                new_examples["prompt"].append (prompt)
        return new_examples

    tokenized_train_dataset = train_dataset.map (
        preprocess_function, batched=True, num_proc=4, remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )
    train_dataset = tokenized_train_dataset.filter (lambda x: len (x['prompt']) > 0)

    tokenized_eval_dataset = eval_dataset.map (
        preprocess_function, batched=True, num_proc=4, remove_columns=eval_dataset.column_names,
        load_from_cache_file=False
    )
    eval_dataset = tokenized_eval_dataset.filter (lambda x: len (x['prompt']) > 0)

    # Build RLOO trainer
    trainer = RLOOTrainer (
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        reward_funcs=reward_model,
        reward_processing_classes=[tokenizer],  # Configure the model consistently with its training checkpoint.
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Train
    if training_args.do_train:
        if is_main_process:
            logger.info ("*** Train ***")
        trainer.train ()
        if is_main_process:
            trainer.save_model (training_args.output_dir)


if __name__ == "__main__":
    main ()
