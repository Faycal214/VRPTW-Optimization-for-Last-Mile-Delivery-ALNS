"""ALNS acceptance criteria for VRPTW.

This module implements simulated annealing style acceptance for the ALNS loop.
It decides whether a candidate solution replaces the current one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import random

from core.model import Solution


@dataclass
class AcceptanceState:
    temperature: float
    initial_temperature: float
    cooling_rate: float
    min_temperature: float = 1e-6

    def cool(self) -> None:
        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)


def simulated_annealing_accept(
    current_objective: float,
    candidate_objective: float,
    temperature: float,
    rng: Optional[random.Random] = None,
) -> bool:
    """Return True if the candidate should be accepted.

    - Always accept improving candidates.
    - Otherwise accept with probability exp(-(delta)/T).
    """
    rng = rng or random.Random()

    if candidate_objective <= current_objective:
        return True

    if temperature <= 0:
        return False

    delta = candidate_objective - current_objective
    acceptance_probability = math.exp(-delta / temperature)
    return rng.random() < acceptance_probability


def better_solution_accept(current: Solution, candidate: Solution) -> bool:
    """Simple deterministic acceptance based on objective only."""
    return candidate.objective <= current.objective


def accept_candidate(
    current: Solution,
    candidate: Solution,
    state: AcceptanceState,
    rng: Optional[random.Random] = None,
) -> bool:
    """Generic acceptance dispatcher for ALNS."""
    return simulated_annealing_accept(
        current_objective=current.objective,
        candidate_objective=candidate.objective,
        temperature=state.temperature,
        rng=rng,
    )


def initialize_acceptance_state(initial_temperature: float = 100.0, cooling_rate: float = 0.995) -> AcceptanceState:
    """Create a default acceptance state."""
    return AcceptanceState(
        temperature=initial_temperature,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
    )


__all__ = [
    "AcceptanceState",
    "simulated_annealing_accept",
    "better_solution_accept",
    "accept_candidate",
    "initialize_acceptance_state",
]
