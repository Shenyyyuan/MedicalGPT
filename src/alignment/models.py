"""Small explicit HF adapter; imports heavyweight dependencies only on demand."""
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, actor, hidden_size):
        super().__init__()
        self.actor = actor
        self.value_head = nn.Linear(hidden_size,1,dtype=torch.float32)

    def score(self, ids, prompt_length):
        """Return log pi(a_t|s_t), V(s_t), entropy for generated tokens only.

        Position prompt_length-1 predicts the first completion token. Padding,
        prompt and tool-observation tokens are never included in this objective.
        """
        if ids.ndim != 1 or not 1 <= prompt_length < ids.numel():
            raise ValueError('invalid prompt/completion boundary')
        out = self.actor(input_ids=ids[None,:], attention_mask=torch.ones_like(ids)[None,:],
            output_hidden_states=True, use_cache=False)
        logits = out.logits[0,prompt_length-1:-1].float()
        log_probs = logits.log_softmax(-1)
        logp = log_probs.gather(-1, ids[prompt_length:,None]).squeeze(-1)
        hidden = out.hidden_states[-1][0,prompt_length-1:-1].float()
        values = self.value_head(hidden).squeeze(-1)
        entropy = -(log_probs.exp()*log_probs).sum(-1)
        return logp, values, entropy


def load_actor(checkpoint, device='cpu', lora_rank=0):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
            torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32)
    except ValueError:
        # Qwen3.5 multimodal checkpoints expose their language outputs via this auto class.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(checkpoint, local_files_only=True,
            torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if lora_rank:
        from peft import get_peft_model, LoraConfig
        model = get_peft_model(model, LoraConfig(r=lora_rank,lora_alpha=2*lora_rank,
            lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
    config = getattr(model.config,'text_config',model.config)
    model = ActorCritic(model,config.hidden_size).to(device)
    # Keep training mode for gradient checkpointing, but disable dropout so old
    # and new likelihoods describe the same distribution in PPO.
    for module in model.modules():
        if isinstance(module,nn.Dropout): module.p=0.
    if hasattr(model.actor,'gradient_checkpointing_enable'):
        model.actor.gradient_checkpointing_enable()
        if hasattr(model.actor,'enable_input_require_grads'): model.actor.enable_input_require_grads()
    model.train()
    return model, tokenizer


def encode_pair(tokenizer, messages, answer, max_length, tools=None):
    kwargs = {'tools':tools} if tools else {}
    prompt = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,**kwargs)
    full = tokenizer.apply_chat_template(messages+[{'role':'assistant','content':answer}],
        tokenize=True,add_generation_prompt=False,**kwargs)
    if full[:len(prompt)] != prompt:
        raise ValueError('chat template is not prefix-stable; use model-specific assistant masks')
    if len(full) > max_length or len(full) <= len(prompt):
        raise ValueError('response empty or exceeds max_length; truncation is forbidden')
    return torch.tensor(full,dtype=torch.long), len(prompt)
