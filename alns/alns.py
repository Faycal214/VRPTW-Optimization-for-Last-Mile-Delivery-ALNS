"""Full ALNS driver with learned operator selection for VRPTW."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import random

from alns.acceptance import AcceptanceState, accept_candidate, initialize_acceptance_state
from alns.destroy import destroy_solution
from alns.repair import repair_solution
from alns.weights import LearnedWeights, initialize_learned_weights, update_weights_by_outcome
from core.constraints import check_solution_feasibility
from core.evaluation import compute_distance_matrix, compute_objective
from core.model import Route, Solution
from core.parser import VRPTWInstance


@dataclass
class ALNSConfig:
    max_iterations: int = 100
    min_remove: int = 5
    max_remove: int = 35
    initial_temperature: float = 100.0
    cooling_rate: float = 0.995
    seed: int = 42
    policy_path: str = "outputs/policy/learned_weights.json"
    load_policy: bool = True
    save_policy: bool = True


@dataclass
class ALNSResult:
    best_solution: Solution
    current_solution: Solution
    iterations: int
    acceptance_state: AcceptanceState
    weights: LearnedWeights
    history: List[float]


def _solution_to_routes(solution: Solution) -> List[List[int]]:
    return [route.path for route in solution.active_routes()]


def _evaluate_solution(solution: Solution, instance: VRPTWInstance, distance_matrix: List[List[float]]) -> Tuple[bool, float]:
    routes = _solution_to_routes(solution)
    feasibility = check_solution_feasibility(routes, instance, distance_matrix)

    if not feasibility.feasible:
        solution.feasible = False
        solution.objective = float("inf")
        return False, float("inf")

    objective, components = compute_objective(routes, instance, distance_matrix)
    solution.objective = objective
    solution.feasible = True
    solution.total_distance = components["total_distance"]
    solution.total_time = components["total_time"]
    solution.vehicles_used = int(components["vehicles_used"])
    solution.load_variance = components["load_variance"]
    solution.spatial_variance = components["spatial_variance"]
    return True, objective


def _initialize_current_solution(solution: Solution, instance: VRPTWInstance, distance_matrix: List[List[float]]) -> Solution:
    cloned = solution.copy()
    feasible, _ = _evaluate_solution(cloned, instance, distance_matrix)
    if not feasible:
        print("Warning: initial solution is infeasible. ALNS will continue anyway.", flush=True)
    return cloned


def run_alns(
    instance: VRPTWInstance,
    initial_solution: Solution,
    distance_matrix: Optional[List[List[float]]] = None,
    max_iterations: int = 100,
    seed: int = 42,
    config: Optional[ALNSConfig] = None,
) -> Solution:
    rng = random.Random(seed)

    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(instance)

    if config is None:
        config = ALNSConfig(max_iterations=max_iterations, seed=seed)
    else:
        config.max_iterations = max_iterations
        config.seed = seed

    if config.load_policy and Path(config.policy_path).exists():
        weights = LearnedWeights.load(config.policy_path)
    else:
        weights = initialize_learned_weights()

    acceptance = initialize_acceptance_state(
        initial_temperature=config.initial_temperature,
        cooling_rate=config.cooling_rate,
    )

    current = _initialize_current_solution(initial_solution, instance, distance_matrix)
    best = current.copy()
    history: List[float] = [best.objective]

    for _ in range(config.max_iterations):
        destroy_name = weights.sample_destroy(rng)
        repair_name = weights.sample_repair(rng)

        remove_count = rng.randint(config.min_remove, config.max_remove)

        destroy_result = destroy_solution(
            solution=current,
            instance=instance,
            num_remove=remove_count,
            method=destroy_name,
            distance_matrix=distance_matrix,
            rng=rng,
        )

        repair_result = repair_solution(
            solution=destroy_result.partial_solution,
            removed_customers=destroy_result.removed_customers,
            instance=instance,
            method=repair_name,
            distance_matrix=distance_matrix,
            rng=rng,
        )

        candidate = repair_result.repaired_solution
        feasible, candidate_objective = _evaluate_solution(candidate, instance, distance_matrix)

        if not feasible:
            weights.penalize(destroy_name, repair_name, penalty=0.15)
            acceptance.cool()
            history.append(best.objective)
            continue

        current_objective = current.objective
        best_objective = best.objective

        accepted = accept_candidate(current, candidate, acceptance, rng=rng)
        if accepted:
            current = candidate.copy()
            current.objective = candidate_objective
            current.feasible = True

        if candidate_objective < best.objective:
            best = candidate.copy()
            best.objective = candidate_objective
            best.feasible = True

        update_weights_by_outcome(
            weights=weights,
            destroy_name=destroy_name,
            repair_name=repair_name,
            current_objective=current_objective,
            candidate_objective=candidate_objective,
            best_objective=best_objective,
        )

        acceptance.cool()
        history.append(best.objective)

    best.metadata["alns_history"] = history
    best.metadata["operator_weights"] = weights.report()
    best.metadata["iterations"] = config.max_iterations
    best.metadata["mode"] = "learned_alns"

    if config.save_policy:
        weights.save(config.policy_path)

    return best


__all__ = [
    "ALNSConfig",
    "ALNSResult",
    "run_alns",
]