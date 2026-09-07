"""Numerically stable objectives with explicit response-token masks.

References: PPO https://arxiv.org/abs/1707.06347,
GAE https://arxiv.org/abs/1506.02438, DPO https://arxiv.org/abs/2305.18290.
Tensor shapes are [batch, time], except sequence log probabilities [batch].
"""
import torch
import torch.nn.functional as F


def masked_mean(values, mask):
    if values.shape != mask.shape or not mask.any():
        raise ValueError('mask must match values and contain supervised tokens')
    selected = values[mask.bool()]
    if not torch.isfinite(selected).all():
        raise FloatingPointError('non-finite objective')
    return selected.mean()


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    """Use SUM of completion log probabilities, as in the original DPO objective."""
    if beta <= 0:
        raise ValueError('beta must be positive')
    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected).detach())
    if not torch.isfinite(margin).all():
        raise FloatingPointError('non-finite DPO margin')
    return -F.logsigmoid(margin).mean(), margin.detach()


def gae(rewards, values, terminated, bootstrap=0., gamma=1., lam=.95):
    """Single episode over sampled tokens; tool observations are not policy actions.

    A truncated episode bootstraps V(next_state). A true terminal does not.
    Values/rewards are detached because PPO treats rollout targets as constants.
    """
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('invalid discount')
    if rewards.ndim != 1 or rewards.shape != values.shape or rewards.shape != terminated.shape or not rewards.numel():
        raise ValueError('expected equally sized nonempty episode vectors')
    with torch.no_grad():
        advantages = torch.zeros_like(rewards)
        carry = torch.zeros((), device=rewards.device)
        next_value = torch.as_tensor(bootstrap, device=rewards.device)
        for t in reversed(range(len(rewards))):
            live = 1 - terminated[t].float()
            delta = rewards[t] + gamma * live * next_value - values[t]
            carry = delta + gamma * lam * live * carry
            advantages[t] = carry
            next_value = values[t]
        return advantages, advantages + values


def ppo_loss(new_logp, old_logp, advantages, values, old_values, returns, mask,
             clip=.2, value_clip=.2, value_coef=.5, entropy=None, entropy_coef=0.):
    """Clipped policy objective plus clipped value regression (Schulman et al.)."""
    if not 0 < clip < 1 or value_clip < 0 or value_coef < 0:
        raise ValueError('invalid PPO coefficients')
    log_ratio = new_logp - old_logp.detach()
    # Never silently clamp likelihood ratios: large drift must trigger diagnostics.
    ratio = log_ratio.exp()
    policy = -masked_mean(torch.minimum(ratio * advantages.detach(),
        ratio.clamp(1-clip, 1+clip) * advantages.detach()), mask)
    clipped_values = old_values.detach() + (values-old_values.detach()).clamp(-value_clip, value_clip)
    vloss = .5 * masked_mean(torch.maximum((values-returns.detach()).square(),
        (clipped_values-returns.detach()).square()), mask)
    ent = masked_mean(entropy, mask) if entropy is not None else policy.new_tensor(0.)
    loss = policy + value_coef * vloss - entropy_coef * ent
    metrics = {'policy_loss':policy.detach().item(), 'value_loss':vloss.detach().item(),
        'approx_kl':masked_mean((ratio-1)-log_ratio,mask).detach().item(),
        'clip_fraction':masked_mean(((ratio-1).abs()>clip).float(),mask).item()}
    return loss, metrics
