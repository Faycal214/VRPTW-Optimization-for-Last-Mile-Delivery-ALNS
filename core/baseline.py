"""Compact and strong VRPTW baseline (ALNS-ready)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from core.constraints import check_solution_feasibility, fast_route_check
from core.evaluation import compute_distance_matrix, compute_objective
from core.parser import VRPTWInstance


@dataclass
class BaselineSolution:
    routes: List[List[int]]
    feasible: bool
    objective: float
    total_distance: float
    total_time: float
    vehicles_used: int
    notes: List[str]


# -----------------------------------------------------------------------------
# FAST INSERTION UTILITIES
# -----------------------------------------------------------------------------

def _try_insertion(
    route: List[int],
    customer_id: int,
    instance: VRPTWInstance,
    distance_matrix: List[List[float]],
) -> Tuple[int, float] | None:
    """Return best feasible insertion (position, cost)."""

    best_pos = None
    best_cost = float("inf")

    for pos in range(1, len(route)):
        trial = route[:pos] + [customer_id] + route[pos:]

        if not fast_route_check(trial, instance, distance_matrix):
            continue

        prev_node = route[pos - 1]
        next_node = route[pos]

        cost = (
            distance_matrix[prev_node][customer_id]
            + distance_matrix[customer_id][next_node]
            - distance_matrix[prev_node][next_node]
        )

        if cost < best_cost:
            best_cost = cost
            best_pos = pos

    if best_pos is None:
        return None

    return best_pos, best_cost


# -----------------------------------------------------------------------------
# MAIN BASELINE
# -----------------------------------------------------------------------------

def greedy_sequential_insertion(instance: VRPTWInstance) -> List[List[int]]:
    """Build routes sequentially (strong + compact)."""

    distance_matrix = compute_distance_matrix(instance)

    unserved = set(range(1, instance.num_nodes))
    routes: List[List[int]] = []

    # Sort by urgency (tight windows first)
    ordered = sorted(
        list(unserved),
        key=lambda cid: (
            instance.all_nodes[cid].due_date,
            instance.all_nodes[cid].ready_time,
        ),
    )

    for seed in ordered:
        if seed not in unserved:
            continue

        # Try to start a valid route
        route = [0, seed, 0]

        if not fast_route_check(route, instance, distance_matrix):
            continue  # skip impossible seeds

        unserved.remove(seed)

        # Grow route greedily
        while True:
            best_customer = None
            best_pos = None
            best_cost = float("inf")

            for cid in unserved:
                result = _try_insertion(route, cid, instance, distance_matrix)
                if result is None:
                    continue

                pos, cost = result
                if cost < best_cost:
                    best_cost = cost
                    best_customer = cid
                    best_pos = pos

            if best_customer is None:
                break

            route.insert(best_pos, best_customer)
            unserved.remove(best_customer)

        routes.append(route)

        # Respect fleet limit
        if len(routes) >= instance.num_vehicles:
            break

    # --- Fallback (rare but safe) ---
    for cid in list(unserved):
        single = [0, cid, 0]
        if fast_route_check(single, instance, distance_matrix):
            routes.append(single)
            unserved.remove(cid)

    return routes


# -----------------------------------------------------------------------------
# BUILD + SCORE
# -----------------------------------------------------------------------------

def build_baseline_solution(instance: VRPTWInstance) -> BaselineSolution:
    distance_matrix = compute_distance_matrix(instance)
    routes = greedy_sequential_insertion(instance)

    feasibility = check_solution_feasibility(
        routes, instance, distance_matrix, check_global=True
    )

    if feasibility.feasible:
        objective, comp = compute_objective(routes, instance, distance_matrix)

        return BaselineSolution(
            routes=routes,
            feasible=True,
            objective=objective,
            total_distance=comp["total_distance"],
            total_time=comp["total_time"],
            vehicles_used=int(comp["vehicles_used"]),
            notes=[],
        )

    return BaselineSolution(
        routes=routes,
        feasible=False,
        objective=float("inf"),
        total_distance=feasibility.total_distance,
        total_time=feasibility.total_time,
        vehicles_used=feasibility.vehicles_used,
        notes=feasibility.errors,
    )


__all__ = [
    "BaselineSolution",
    "greedy_sequential_insertion",
    "build_baseline_solution",
]