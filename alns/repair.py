"""Lightweight ALNS repair operators for VRPTW.

This version is intentionally compact and fast.
It avoids expensive full-solution feasibility checks inside insertion search.

Implemented operators:
- greedy insertion
- regret-2 insertion (mapped to the same fast heuristic)
- regret-3 insertion (mapped to the same fast heuristic)
- generic repair dispatcher

The goal is to keep the solver responsive on a low-resource machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import random

from core.model import InsertionMove, Route, Solution
from core.parser import VRPTWInstance
from core.evaluation import compute_distance_matrix


@dataclass
class RepairResult:
    repaired_solution: Solution
    inserted_customers: List[int]
    insertion_moves: List[InsertionMove]
    success: bool
    errors: List[str]


def _copy_solution(solution: Solution) -> Solution:
    return solution.copy()


def _ensure_route_ends(route: Route) -> None:
    if not route.path:
        route.path = [0, 0]
    if route.path[0] != 0:
        route.path.insert(0, 0)
    if route.path[-1] != 0:
        route.path.append(0)
    if len(route.path) == 1:
        route.path.append(0)


def _route_is_feasible_fast(path: Sequence[int], instance: VRPTWInstance, distance_matrix: List[List[float]]) -> bool:
    """Fast single-route feasibility check.

    Checks only:
    - route endpoints
    - capacity
    - time windows
    - depot return
    """
    if not path or path[0] != 0 or path[-1] != 0:
        return False

    time = 0.0
    load = 0.0

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        time += distance_matrix[a][b]

        if b == 0:
            continue

        node = instance.all_nodes[b]
        load += node.demand
        if load > instance.capacity:
            return False

        if time > node.due_date:
            return False

        if time < node.ready_time:
            time = node.ready_time
        time += node.service_time

    return time <= instance.depot.due_date


def _candidate_positions(path: Sequence[int]) -> List[int]:
    """Return a small set of insertion positions to keep the search cheap."""
    n = len(path)
    if n <= 2:
        return [1]

    positions = {1, n - 1}
    if n > 4:
        positions.add(n // 2)

    return sorted(p for p in positions if 1 <= p <= n - 1)


def _insertion_cost(route: Sequence[int], customer_id: int, position: int, distance_matrix: List[List[float]]) -> float:
    prev_node = route[position - 1]
    next_node = route[position]
    added = distance_matrix[prev_node][customer_id] + distance_matrix[customer_id][next_node]
    removed = distance_matrix[prev_node][next_node]
    return added - removed


def _try_insert_into_route(
    route: Route,
    customer_id: int,
    instance: VRPTWInstance,
    distance_matrix: List[List[float]],
) -> List[Tuple[int, float]]:
    """Return feasible insertion positions and costs for one route."""
    _ensure_route_ends(route)
    feasible_positions: List[Tuple[int, float]] = []

    for position in _candidate_positions(route.path):
        trial = route.path[:position] + [customer_id] + route.path[position:]
        if _route_is_feasible_fast(trial, instance, distance_matrix):
            delta = _insertion_cost(route.path, customer_id, position, distance_matrix)
            feasible_positions.append((position, delta))

    return feasible_positions


def _insert_customer_at(route: Route, customer_id: int, position: int) -> None:
    _ensure_route_ends(route)
    route.path.insert(position, customer_id)


def _collect_routes_with_indices(solution: Solution) -> List[Tuple[int, Route]]:
    return [(idx, route) for idx, route in enumerate(solution.routes)]


def _new_empty_route() -> Route:
    return Route(path=[0, 0])


def _make_partial_solution(solution: Solution) -> Solution:
    partial = _copy_solution(solution)
    for route in partial.routes:
        _ensure_route_ends(route)
    return partial


def _ordered_customers(removed_customers: Sequence[int], instance: VRPTWInstance) -> List[int]:
    """Sort customers by urgency so tighter instances are handled first."""
    return sorted(
        list(dict.fromkeys(int(c) for c in removed_customers)),
        key=lambda cid: (
            instance.all_nodes[cid].due_date,
            instance.all_nodes[cid].ready_time,
            -instance.all_nodes[cid].demand,
        ),
    )


def greedy_insertion(
    solution: Solution,
    removed_customers: Sequence[int],
    instance: VRPTWInstance,
    distance_matrix: Optional[List[List[float]]] = None,
) -> RepairResult:
    """Insert customers one by one using the cheapest feasible placement."""
    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(instance)

    partial = _make_partial_solution(solution)
    pending = _ordered_customers(removed_customers, instance)
    inserted_customers: List[int] = []
    insertion_moves: List[InsertionMove] = []
    errors: List[str] = []

    for customer_id in pending:
        best_choice: Optional[Tuple[int, int, float]] = None  # route_index, position, delta

        for route_index, route in _collect_routes_with_indices(partial):
            feasible_positions = _try_insert_into_route(route, customer_id, instance, distance_matrix)
            for position, delta in feasible_positions:
                if best_choice is None or delta < best_choice[2]:
                    best_choice = (route_index, position, delta)

        if best_choice is None:
            new_route = _new_empty_route()
            feasible_positions = _try_insert_into_route(new_route, customer_id, instance, distance_matrix)
            if not feasible_positions:
                errors.append(f"Customer {customer_id} could not be inserted feasibly.")
                continue

            position, delta = min(feasible_positions, key=lambda x: x[1])
            new_route.path.insert(position, customer_id)
            partial.routes.append(new_route)
            inserted_customers.append(customer_id)
            insertion_moves.append(
                InsertionMove(
                    customer_id=customer_id,
                    route_index=len(partial.routes) - 1,
                    position=position,
                    delta_cost=delta,
                )
            )
            continue

        route_index, position, delta = best_choice
        _insert_customer_at(partial.routes[route_index], customer_id, position)
        inserted_customers.append(customer_id)
        insertion_moves.append(
            InsertionMove(customer_id=customer_id, route_index=route_index, position=position, delta_cost=delta)
        )

    partial.unserved_customers = [c for c in partial.unserved_customers if c not in inserted_customers]
    return RepairResult(
        repaired_solution=partial,
        inserted_customers=inserted_customers,
        insertion_moves=insertion_moves,
        success=len(errors) == 0,
        errors=errors,
    )


def regret_k_insertion(
    solution: Solution,
    removed_customers: Sequence[int],
    instance: VRPTWInstance,
    k: int = 2,
    distance_matrix: Optional[List[List[float]]] = None,
    rng: Optional[random.Random] = None,
) -> RepairResult:
    """Fast regret-k placeholder.

    For performance on low-resource machines, this uses the same fast greedy
    insertion logic. The interface remains compatible with the full ALNS stack.
    """
    return greedy_insertion(solution, removed_customers, instance, distance_matrix)


def regret_2_insertion(
    solution: Solution,
    removed_customers: Sequence[int],
    instance: VRPTWInstance,
    distance_matrix: Optional[List[List[float]]] = None,
) -> RepairResult:
    return greedy_insertion(solution, removed_customers, instance, distance_matrix)


def regret_3_insertion(
    solution: Solution,
    removed_customers: Sequence[int],
    instance: VRPTWInstance,
    distance_matrix: Optional[List[List[float]]] = None,
    rng: Optional[random.Random] = None,
) -> RepairResult:
    return greedy_insertion(solution, removed_customers, instance, distance_matrix)


def repair_solution(
    solution: Solution,
    removed_customers: Sequence[int],
    instance: VRPTWInstance,
    method: str = "greedy",
    distance_matrix: Optional[List[List[float]]] = None,
    rng: Optional[random.Random] = None,
) -> RepairResult:
    """Generic repair dispatcher."""
    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(instance)

    method = method.lower().strip()
    if method == "greedy":
        return greedy_insertion(solution, removed_customers, instance, distance_matrix)
    if method in {"regret2", "regret_2", "r2"}:
        return regret_2_insertion(solution, removed_customers, instance, distance_matrix)
    if method in {"regret3", "regret_3", "r3"}:
        return regret_3_insertion(solution, removed_customers, instance, distance_matrix, rng=rng)

    raise ValueError(f"Unknown repair method: {method}")


__all__ = [
    "RepairResult",
    "greedy_insertion",
    "regret_k_insertion",
    "regret_2_insertion",
    "regret_3_insertion",
    "repair_solution",
]