"""Versioned rollout storage, not an off-policy replay buffer.

PPO reuses the current behavior-policy batch for a bounded number of epochs.
It must discard that batch before sampling from the next policy version.
"""
import random
from dataclasses import dataclass
import torch


@dataclass
class Rollout:
    input_ids: torch.Tensor
    prompt_length: int
    old_logp: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    policy_version: int


class RolloutBuffer:
    def __init__(self, policy_version, capacity=256):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.policy_version, self.capacity = policy_version, capacity
        self.rows, self.sealed = [], False
        self.consumed = False

    def add(self, row):
        if self.sealed or row.policy_version != self.policy_version:
            raise ValueError('sealed or stale-policy rollout')
        if len(self.rows) >= self.capacity:
            raise OverflowError('rollout capacity exceeded; refusing silent eviction')
        n = row.input_ids.numel() - row.prompt_length
        if n < 1 or row.prompt_length < 1 or any(x.numel()!=n for x in
                [row.old_logp,row.old_values,row.advantages,row.returns]):
            raise ValueError('invalid response alignment')
        self.rows.append(row)

    def seal(self, normalize=True):
        if self.sealed or not self.rows:
            raise ValueError('empty or already sealed buffer')
        if normalize:
            all_adv = torch.cat([x.advantages for x in self.rows])
            mean, std = all_adv.mean(), all_adv.std(unbiased=False)
            for row in self.rows:
                row.advantages = (row.advantages-mean)/(std+1e-8)
        self.sealed = True

    def batches(self, size, seed):
        if not self.sealed or size < 1:
            raise ValueError('seal buffer and choose positive batch size')
        indices = list(range(len(self.rows)))
        random.Random(seed).shuffle(indices)
        for i in range(0,len(indices),size):
            yield [self.rows[j] for j in indices[i:i+size]]
