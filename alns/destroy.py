"""ALNS destroy operators for VRPTW.

This module removes customers from a solution to create a partial solution that
will later be repaired by insertion heuristics.

Implemented operators:
- random removal
- worst removal
- related removal
- route removal

All operators work on the `core.model.Solution` and `core.model.Route` classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

from core.model import RemovalMove, Route, Solution
from core.parser import VRPTWInstance
from core.evaluation import compute_distance_matrix


@dataclass
class DestroyResult:
    partial_solution: Solution
    removed_customers: List[int]
    removed_moves: List[RemovalMove]


def _customer_positions(solution: Solution) -> List[Tuple[int, int, int]]:
    """Return all customer positions as (route_index, position_in_route, customer_id)."""
    positions: List[Tuple[int, int, int]] = []
    for r_idx, route in enumerate(solution.routes):
        if len(route.path) <= 2:
            continue
        for pos in range(1, len(route.path) - 1):
            positions.append((r_idx, pos, route.path[pos]))
    return positions


def _copy_solution(solution: Solution) -> Solution:
    return solution.copy()


def _remove_customer_from_route(route: Route, position: int) -> int:
    """Remove a customer from a route path and return the removed id."""
    return route.path.pop(position)


def _cleanup_empty_routes(solution: Solution) -> None:
    """Normalize empty routes to [0, 0] for later repair logic."""
    for route in solution.routes:
        if len(route.path) < 2:
            route.path = [0, 0]
        if route.path == [0] or route.path == []:
            route.path = [0, 0]


def random_removal(solution: Solution, num_remove: int, rng: Optional[random.Random] = None) -> DestroyResult:
    """Remove a random subset of customers from the solution."""
    rng = rng or random.Random()
    partial = _copy_solution(solution)
    positions = _customer_positions(partial)

    if not positions:
        return DestroyResult(partial, [], [])

    num_remove = max(0, min(num_remove, len(positions)))
    chosen = rng.sample(positions, k=num_remove)

    removed_customers: List[int] = []
    removed_moves: List[RemovalMove] = []

    # Reverse sort so indices stay valid while popping.
    chosen = sorted(chosen, key=lambda x: (x[0], x[1]), reverse=True)

    for r_idx, pos, cust_id in chosen:
        route = partial.routes[r_idx]
        if pos >= len(route.path) or route.path[pos] != cust_id:
            try:
                pos = route.path.index(cust_id)
            except ValueError:
                continue

        if 0 < pos < len(route.path) - 1:
            removed = _remove_customer_from_route(route, pos)
            removed_customers.append(removed)
            removed_moves.append(RemovalMove(customer_id=removed, route_index=r_idx, position=pos))

    _cleanup_empty_routes(partial)
    partial.unserved_customers = sorted(set(partial.unserved_customers + removed_customers))
    return DestroyResult(partial, removed_customers, removed_moves)


def route_removal(solution: Solution, num_routes_to_remove: int, rng: Optional[random.Random] = None) -> DestroyResult:
    """Remove entire routes and return all their customers to the unserved pool."""
    rng = rng or random.Random()
    partial = _copy_solution(solution)
    active_indices = [i for i, r in enumerate(partial.routes) if len(r.path) > 2]

    if not active_indices:
        return DestroyResult(partial, [], [])

    num_routes_to_remove = max(0, min(num_routes_to_remove, len(active_indices)))
    chosen = rng.sample(active_indices, k=num_routes_to_remove)

    removed_customers: List[int] = []
    removed_moves: List[RemovalMove] = []

    for r_idx in chosen:
        route = partial.routes[r_idx]
        for pos in range(1, len(route.path) - 1):
            cust = route.path[pos]
            removed_customers.append(cust)
            removed_moves.append(RemovalMove(customer_id=cust, route_index=r_idx, position=pos))
        route.path = [0, 0]

    _cleanup_empty_routes(partial)
    partial.unserved_customers = sorted(set(partial.unserved_customers + removed_customers))
    return DestroyResult(partial, removed_customers, removed_moves)


def worst_removal(
    solution: Solution,
    num_remove: int,
    distance_matrix: List[List[float]],
    rng: Optional[random.Random] = None,
) -> DestroyResult:
    """Remove customers that contribute the largest marginal distance cost."""
    rng = rng or random.Random()
    partial = _copy_solution(solution)
    candidates: List[Tuple[float, int, int, int]] = []

    for r_idx, route in enumerate(partial.routes):
        path = route.path
        if len(path) <= 2:
            continue
        for pos in range(1, len(path) - 1):
            prev_node = path[pos - 1]
            cust = path[pos]
            next_node = path[pos + 1]
            saving = (
                distance_matrix[prev_node][cust]
                + distance_matrix[cust][next_node]
                - distance_matrix[prev_node][next_node]
            )
            candidates.append((saving, r_idx, pos, cust))

    if not candidates:
        return DestroyResult(partial, [], [])

    candidates.sort(key=lambda x: x[0], reverse=True)
    pool_size = min(len(candidates), max(num_remove * 3, num_remove))
    pool = candidates[:pool_size]
    chosen = rng.sample(pool, k=min(num_remove, len(pool)))
    chosen = sorted(chosen, key=lambda x: (x[1], x[2]), reverse=True)

    removed_customers: List[int] = []
    removed_moves: List[RemovalMove] = []

    for _, r_idx, pos, cust_id in chosen:
        route = partial.routes[r_idx]
        if pos >= len(route.path) or route.path[pos] != cust_id:
            try:
                pos = route.path.index(cust_id)
            except ValueError:
                continue

        if 0 < pos < len(route.path) - 1:
            removed = _remove_customer_from_route(route, pos)
            removed_customers.append(removed)
            removed_moves.append(RemovalMove(customer_id=removed, route_index=r_idx, position=pos))

    _cleanup_empty_routes(partial)
    partial.unserved_customers = sorted(set(partial.unserved_customers + removed_customers))
    return DestroyResult(partial, removed_customers, removed_moves)


def related_removal(
    solution: Solution,
    num_remove: int,
    instance: VRPTWInstance,
    distance_matrix: List[List[float]],
    rng: Optional[random.Random] = None,
    seed_customer: Optional[int] = None,
    alpha: float = 0.6,
) -> DestroyResult:
    """Remove customers that are geographically and temporally related."""
    rng = rng or random.Random()
    partial = _copy_solution(solution)
    positions = _customer_positions(partial)

    if not positions:
        return DestroyResult(partial, [], [])

    if seed_customer is None:
        _, _, seed_customer = rng.choice(positions)

    seed_node = instance.all_nodes[seed_customer]

    scored: List[Tuple[float, int, int, int]] = []
    for r_idx, pos, cust_id in positions:
        node = instance.all_nodes[cust_id]
        spatial = distance_matrix[seed_customer][cust_id]
        tw_gap = abs(seed_node.ready_time - node.ready_time) + abs(seed_node.due_date - node.due_date)
        score = alpha * spatial + (1.0 - alpha) * tw_gap
        scored.append((score, r_idx, pos, cust_id))

    scored.sort(key=lambda x: x[0])
    pool_size = min(len(scored), max(num_remove * 3, num_remove))
    pool = scored[:pool_size]
    chosen = rng.sample(pool, k=min(num_remove, len(pool)))
    chosen = sorted(chosen, key=lambda x: (x[1], x[2]), reverse=True)

    removed_customers: List[int] = []
    removed_moves: List[RemovalMove] = []

    for _, r_idx, pos, cust_id in chosen:
        route = partial.routes[r_idx]
        if pos >= len(route.path) or route.path[pos] != cust_id:
            try:
                pos = route.path.index(cust_id)
            except ValueError:
                continue

        if 0 < pos < len(route.path) - 1:
            removed = _remove_customer_from_route(route, pos)
            removed_customers.append(removed)
            removed_moves.append(RemovalMove(customer_id=removed, route_index=r_idx, position=pos))

    _cleanup_empty_routes(partial)
    partial.unserved_customers = sorted(set(partial.unserved_customers + removed_customers))
    return DestroyResult(partial, removed_customers, removed_moves)


def destroy_solution(
    solution: Solution,
    instance: VRPTWInstance,
    num_remove: int,
    method: str = "related",
    distance_matrix: Optional[List[List[float]]] = None,
    rng: Optional[random.Random] = None,
) -> DestroyResult:
    """Generic destroy dispatcher."""
    rng = rng or random.Random()

    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(instance)

    method = method.lower().strip()

    if method == "random":
        return random_removal(solution, num_remove=num_remove, rng=rng)
    if method == "worst":
        return worst_removal(solution, num_remove=num_remove, distance_matrix=distance_matrix, rng=rng)
    if method == "related":
        return related_removal(
            solution,
            num_remove=num_remove,
            instance=instance,
            distance_matrix=distance_matrix,
            rng=rng,
        )
    if method == "route":
        return route_removal(solution, num_routes_to_remove=max(1, num_remove // 5), rng=rng)

    raise ValueError(f"Unknown destroy method: {method}")


__all__ = [
    "DestroyResult",
    "random_removal",
    "worst_removal",
    "related_removal",
    "route_removal",
    "destroy_solution",
]