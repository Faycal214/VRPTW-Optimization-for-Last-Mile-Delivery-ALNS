"""Fast VRPTW feasibility checking (ALNS-friendly).

Key improvements:
- Lightweight route-level checks for ALNS
- Optional global checks (only at final evaluation)
- Early stopping on infeasibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Set

from core.parser import VRPTWInstance


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class RouteCheckResult:
    feasible: bool
    distance: float = 0.0
    total_time: float = 0.0
    load: float = 0.0


@dataclass
class FeasibilityResult:
    feasible: bool
    errors: List[str]
    vehicles_used: int
    total_distance: float
    total_time: float

def routes_from_submission_format(routes_payload) -> List[List[int]]:
    """Normalize a routes payload into a list of integer routes.

    Accepted shapes:
    - [[0, 1, 2, 0], [0, 3, 4, 0]]
    - [{"path": [0, 1, 2, 0]}, {"path": [0, 3, 4, 0]}]
    """
    normalized: List[List[int]] = []

    if routes_payload is None:
        return normalized

    for item in routes_payload:
        if isinstance(item, dict) and "path" in item:
            normalized.append([int(x) for x in item["path"]])
        elif isinstance(item, (list, tuple)):
            normalized.append([int(x) for x in item])
        else:
            raise ValueError(f"Unsupported route format: {item}")

    return normalized


# -----------------------------------------------------------------------------
# FAST ROUTE CHECK (used inside ALNS)
# -----------------------------------------------------------------------------

def check_route_feasible(
    route: Sequence[int],
    instance: VRPTWInstance,
    distance_matrix: List[List[float]],
) -> RouteCheckResult:
    """Fast feasibility check for ONE route only.

    Stops immediately when infeasible.
    """

    if len(route) < 2:
        return RouteCheckResult(False)

    time = 0.0
    load = 0.0
    distance = 0.0

    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]

        travel = distance_matrix[a][b]
        distance += travel
        time += travel

        if b == 0:
            continue

        customer = instance.all_nodes[b]

        # Time window
        if time > customer.due_date:
            return RouteCheckResult(False)

        # Wait if early
        if time < customer.ready_time:
            time = customer.ready_time

        time += customer.service_time

        # Capacity
        load += customer.demand
        if load > instance.capacity:
            return RouteCheckResult(False)

    # Depot return constraint
    if time > instance.depot.due_date:
        return RouteCheckResult(False)

    return RouteCheckResult(
        feasible=True,
        distance=distance,
        total_time=time,
        load=load,
    )


# -----------------------------------------------------------------------------
# GLOBAL CHECK (used ONLY at final evaluation)
# -----------------------------------------------------------------------------

def check_solution_feasibility(
    routes: Sequence[Sequence[int]],
    instance: VRPTWInstance,
    distance_matrix: List[List[float]],
    check_global: bool = True,
) -> FeasibilityResult:
    """Full solution check.

    - Fast route validation
    - Optional global constraints (coverage, duplicates)
    """

    total_distance = 0.0
    total_time = 0.0
    errors: List[str] = []

    visited: Set[int] = set()
    duplicates: Set[int] = set()

    vehicles_used = 0

    for route in routes:
        if len(route) < 2:
            continue

        vehicles_used += 1

        result = check_route_feasible(route, instance, distance_matrix)

        if not result.feasible:
            errors.append("Route infeasible")
            continue

        total_distance += result.distance
        total_time += result.total_time

        if check_global:
            for node in route:
                if node == 0:
                    continue
                if node in visited:
                    duplicates.add(node)
                visited.add(node)

    # --- Global checks (ONLY when needed) ---
    if check_global:
        expected = set(range(1, instance.num_nodes))
        missing = expected - visited

        if missing:
            errors.append(f"Missing customers ({len(missing)})")

        if duplicates:
            errors.append(f"Duplicate customers ({len(duplicates)})")

        if vehicles_used > instance.num_vehicles:
            errors.append(
                f"Too many vehicles ({vehicles_used} > {instance.num_vehicles})"
            )

    return FeasibilityResult(
        feasible=len(errors) == 0,
        errors=errors,
        vehicles_used=vehicles_used,
        total_distance=total_distance,
        total_time=total_time,
    )


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def fast_route_check(
    route: Sequence[int],
    instance: VRPTWInstance,
    distance_matrix: List[List[float]],
) -> bool:
    """Ultra-light boolean check (for inner loops)."""
    return check_route_feasible(route, instance, distance_matrix).feasible


__all__ = [
    "RouteCheckResult",
    "FeasibilityResult",
    "routes_from_submission_format",
    "check_route_feasible",
    "check_solution_feasibility",
    "fast_route_check",
]