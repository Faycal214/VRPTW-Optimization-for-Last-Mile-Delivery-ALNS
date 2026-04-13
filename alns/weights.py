"""Learned operator weights for VRPTW ALNS.

This module implements a lightweight bandit-style policy:
- select destroy/repair operators with weighted sampling
- update weights using reward from solution improvement
- save/load learned weights to JSON

This is the first AI layer in the hybrid solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json
import math
import random


DEFAULT_DESTROY_METHODS = ["random", "worst", "related", "route"]
DEFAULT_REPAIR_METHODS = ["greedy", "regret_2", "regret_3"]


@dataclass
class LearnedWeights:
    destroy_weights: Dict[str, float] = field(default_factory=dict)
    repair_weights: Dict[str, float] = field(default_factory=dict)
    destroy_scores: Dict[str, float] = field(default_factory=dict)
    repair_scores: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.2
    min_weight: float = 0.05
    max_weight: float = 50.0

    def __post_init__(self) -> None:
        if not self.destroy_weights:
            self.destroy_weights = {m: 1.0 for m in DEFAULT_DESTROY_METHODS}
        if not self.repair_weights:
            self.repair_weights = {m: 1.0 for m in DEFAULT_REPAIR_METHODS}
        if not self.destroy_scores:
            self.destroy_scores = {m: 0.0 for m in self.destroy_weights}
        if not self.repair_scores:
            self.repair_scores = {m: 0.0 for m in self.repair_weights}

    def _sample(self, weights: Dict[str, float], rng: random.Random) -> str:
        names = list(weights.keys())
        vals = [max(self.min_weight, float(weights[n])) for n in names]
        total = sum(vals)
        if total <= 0:
            return rng.choice(names)
        return rng.choices(names, weights=vals, k=1)[0]

    def sample_destroy(self, rng: random.Random) -> str:
        return self._sample(self.destroy_weights, rng)

    def sample_repair(self, rng: random.Random) -> str:
        return self._sample(self.repair_weights, rng)

    def _update_dict(self, weights: Dict[str, float], scores: Dict[str, float], name: str, reward: float) -> None:
        old_score = scores.get(name, 0.0)
        new_score = (1.0 - self.learning_rate) * old_score + self.learning_rate * reward
        scores[name] = new_score

        # Convert score into a positive weight.
        new_weight = math.exp(new_score)
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        weights[name] = new_weight

    def update(self, destroy_name: str, repair_name: str, reward: float) -> None:
        self._update_dict(self.destroy_weights, self.destroy_scores, destroy_name, reward)
        self._update_dict(self.repair_weights, self.repair_scores, repair_name, reward)

    def penalize(self, destroy_name: str, repair_name: str, penalty: float = 0.2) -> None:
        self.destroy_weights[destroy_name] = max(self.min_weight, self.destroy_weights.get(destroy_name, 1.0) * (1.0 - penalty))
        self.repair_weights[repair_name] = max(self.min_weight, self.repair_weights.get(repair_name, 1.0) * (1.0 - penalty))

    def report(self) -> Dict[str, Dict[str, float]]:
        return {
            "destroy_weights": dict(self.destroy_weights),
            "repair_weights": dict(self.repair_weights),
            "destroy_scores": dict(self.destroy_scores),
            "repair_scores": dict(self.repair_scores),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "destroy_weights": self.destroy_weights,
                    "repair_weights": self.repair_weights,
                    "destroy_scores": self.destroy_scores,
                    "repair_scores": self.repair_scores,
                    "learning_rate": self.learning_rate,
                    "min_weight": self.min_weight,
                    "max_weight": self.max_weight,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedWeights":
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            destroy_weights=data.get("destroy_weights", {}),
            repair_weights=data.get("repair_weights", {}),
            destroy_scores=data.get("destroy_scores", {}),
            repair_scores=data.get("repair_scores", {}),
            learning_rate=float(data.get("learning_rate", 0.2)),
            min_weight=float(data.get("min_weight", 0.05)),
            max_weight=float(data.get("max_weight", 50.0)),
        )


def initialize_learned_weights() -> LearnedWeights:
    return LearnedWeights()


def update_weights_by_outcome(
    weights: LearnedWeights,
    destroy_name: str,
    repair_name: str,
    current_objective: float,
    candidate_objective: float,
    best_objective: float,
) -> None:
    """
    Reward design:
    - strong reward if new global best
    - medium reward if improved current
    - small reward if accepted but not improved
    - penalty otherwise
    """
    if candidate_objective < best_objective:
        reward = 5.0
    elif candidate_objective < current_objective:
        reward = 2.0
    elif candidate_objective == current_objective:
        reward = 0.5
    else:
        reward = -1.0

    weights.update(destroy_name, repair_name, reward)


__all__ = [
    "LearnedWeights",
    "initialize_learned_weights",
    "update_weights_by_outcome",
    "DEFAULT_DESTROY_METHODS",
    "DEFAULT_REPAIR_METHODS",
]