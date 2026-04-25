from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


DESTROY_OPS = ["random", "worst", "related", "route"]
REPAIR_OPS = ["greedy", "regret2", "regret3"]
ACTION_SPACE: List[Tuple[str, str]] = [(d, r) for d in DESTROY_OPS for r in REPAIR_OPS]


class OperatorPolicyNet(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, n_actions: int = len(ACTION_SPACE)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        h = self.encoder(state)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(self, state: torch.Tensor):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy


@dataclass
class Transition:
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    entropy: torch.Tensor