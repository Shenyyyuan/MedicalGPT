"""Optimizer loops independent of HF generation, so core math can be tested offline."""
import torch
from .losses import dpo_loss, ppo_loss


def step(optimizer, parameters, loss, max_norm=1.):
    if not torch.isfinite(loss):
        raise FloatingPointError('non-finite loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def train_dpo_batch(policy, reference, optimizer, pairs, beta=.1):
    pc,pr,rc,rr = [],[],[],[]
    device = next(policy.parameters()).device
    for chosen,rejected,prompt_length in pairs:
        pc.append(policy.score(chosen.to(device),prompt_length)[0].sum())
        pr.append(policy.score(rejected.to(device),prompt_length)[0].sum())
        with torch.no_grad():
            rc.append(reference.score(chosen.to(device),prompt_length)[0].sum())
            rr.append(reference.score(rejected.to(device),prompt_length)[0].sum())
    loss,margin = dpo_loss(*[torch.stack(v) for v in [pc,pr,rc,rr]],beta=beta)
    norm = step(optimizer,policy.parameters(),loss)
    return {'loss':loss.item(),'preference_accuracy':(margin>0).float().mean().item(),'grad_norm':norm}


def train_ppo(policy, optimizer, buffer, epochs=4, batch_size=4, seed=42, target_kl=.03, **loss_options):
    if epochs < 1 or target_kl <= 0:
        raise ValueError('invalid PPO update configuration')
    if buffer.consumed:
        raise ValueError('rollout batch was already consumed by a policy update')
    buffer.consumed = True
    device = next(policy.parameters()).device
    logs = []
    for epoch in range(epochs):
        for batch in buffer.batches(batch_size,seed+epoch):
            losses,stats,weights = [],[],[]
            for row in batch:
                logp,values,entropy = policy.score(row.input_ids.to(device),row.prompt_length)
                tensors = [v.to(device) for v in [row.old_logp,row.advantages,row.old_values,row.returns]]
                old,adv,old_values,returns = tensors
                loss,metrics = ppo_loss(logp,old,adv,values,old_values,returns,
                    torch.ones_like(logp,dtype=torch.bool),entropy=entropy,**loss_options)
                losses.append(loss); stats.append(metrics); weights.append(logp.numel())
            # Token-weighted aggregation avoids overweighting short tool calls.
            merged = {k:sum(s[k]*w for s,w in zip(stats,weights))/sum(weights) for k in stats[0]}
            if merged['approx_kl'] > target_kl:
                return logs+[dict(merged,early_stop=True,epoch=epoch)]
            loss = sum(v*w for v,w in zip(losses,weights))/sum(weights)
            norm = step(optimizer,policy.parameters(),loss)
            logs.append(dict(merged,loss=loss.item(),grad_norm=norm,epoch=epoch,early_stop=False))
    return logs
