"""Adaptive operator weights for ALNS.

This module tracks operator performance and updates selection probabilities
based on solution quality improvements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import math
import random


@dataclass
class OperatorStats:
    score: float = 0.0
    uses: int = 0
    successes: int = 0

    def reward(self, amount: float = 1.0) -> None:
        self.score += amount
        self.successes += 1
        self.uses += 1

    def penalize(self, amount: float = 0.1) -> None:
        self.score = max(0.0, self.score - amount)
        self.uses += 1

    def average_score(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.score / self.uses


@dataclass
class AdaptiveWeights:
    destroy_weights: Dict[str, float] = field(default_factory=dict)
    repair_weights: Dict[str, float] = field(default_factory=dict)
    destroy_stats: Dict[str, OperatorStats] = field(default_factory=dict)
    repair_stats: Dict[str, OperatorStats] = field(default_factory=dict)
    reaction: float = 0.2
    min_weight: float = 0.01
    max_weight: float = 100.0

    def register_destroy_operator(self, name: str, initial_weight: float = 1.0) -> None:
        if name not in self.destroy_weights:
            self.destroy_weights[name] = initial_weight
        if name not in self.destroy_stats:
            self.destroy_stats[name] = OperatorStats()

    def register_repair_operator(self, name: str, initial_weight: float = 1.0) -> None:
        if name not in self.repair_weights:
            self.repair_weights[name] = initial_weight
        if name not in self.repair_stats:
            self.repair_stats[name] = OperatorStats()

    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(self.min_weight, w) for w in weights.values())
        if total <= 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights} if n else {}
        return {k: max(self.min_weight, w) / total for k, w in weights.items()}

    def normalized_destroy_weights(self) -> Dict[str, float]:
        return self._normalize(self.destroy_weights)

    def normalized_repair_weights(self) -> Dict[str, float]:
        return self._normalize(self.repair_weights)

    def sample_destroy(self, rng: random.Random) -> str:
        names = list(self.destroy_weights.keys())
        weights = [max(self.min_weight, self.destroy_weights[n]) for n in names]
        return rng.choices(names, weights=weights, k=1)[0]

    def sample_repair(self, rng: random.Random) -> str:
        names = list(self.repair_weights.keys())
        weights = [max(self.min_weight, self.repair_weights[n]) for n in names]
        return rng.choices(names, weights=weights, k=1)[0]

    def update_destroy(self, name: str, reward: float) -> None:
        if name not in self.destroy_weights:
            self.register_destroy_operator(name)
        self.destroy_stats[name].reward(reward)
        self.destroy_weights[name] = self._clamp(
            (1.0 - self.reaction) * self.destroy_weights[name] + self.reaction * reward
        )

    def update_repair(self, name: str, reward: float) -> None:
        if name not in self.repair_weights:
            self.register_repair_operator(name)
        self.repair_stats[name].reward(reward)
        self.repair_weights[name] = self._clamp(
            (1.0 - self.reaction) * self.repair_weights[name] + self.reaction * reward
        )

    def decay_destroy(self, name: str, penalty: float = 0.05) -> None:
        if name not in self.destroy_weights:
            self.register_destroy_operator(name)
        self.destroy_stats[name].penalize(penalty)
        self.destroy_weights[name] = self._clamp(self.destroy_weights[name] * (1.0 - penalty))

    def decay_repair(self, name: str, penalty: float = 0.05) -> None:
        if name not in self.repair_weights:
            self.register_repair_operator(name)
        self.repair_stats[name].penalize(penalty)
        self.repair_weights[name] = self._clamp(self.repair_weights[name] * (1.0 - penalty))

    def _clamp(self, value: float) -> float:
        return max(self.min_weight, min(self.max_weight, value))

    def report(self) -> Dict[str, Dict[str, float]]:
        return {
            "destroy": {k: float(v) for k, v in self.destroy_weights.items()},
            "repair": {k: float(v) for k, v in self.repair_weights.items()},
        }


def update_weights_by_outcome(
    weights: AdaptiveWeights,
    destroy_name: str,
    repair_name: str,
    current_objective: float,
    candidate_objective: float,
    best_objective: float,
) -> None:
    """Update operator weights based on the outcome of an ALNS iteration.

    Reward scheme:
    - strong reward if candidate improves global best
    - medium reward if candidate improves current solution
    - small reward if accepted but not better
    - penalty otherwise
    """
    if candidate_objective < best_objective:
        reward = 3.0
    elif candidate_objective < current_objective:
        reward = 2.0
    else:
        reward = 0.5

    weights.update_destroy(destroy_name, reward)
    weights.update_repair(repair_name, reward)


def initialize_default_weights() -> AdaptiveWeights:
    """Create a default ALNS operator-weight container and register standard operators."""
    w = AdaptiveWeights(reaction=0.2)
    for name in ["random", "worst", "related", "route"]:
        w.register_destroy_operator(name, initial_weight=1.0)
    for name in ["greedy", "regret_2", "regret_3"]:
        w.register_repair_operator(name, initial_weight=1.0)
    return w


__all__ = [
    "OperatorStats",
    "AdaptiveWeights",
    "update_weights_by_outcome",
    "initialize_default_weights",
]
