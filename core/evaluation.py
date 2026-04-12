"""VRPTW scoring utilities.

This module computes the weighted objective used by the competition evaluation:

    J = 0.35 * T_tot + 0.25 * D + 0.20 * (|R| * 100) + 0.12 * sigma_L^2 + 0.08 * sigma_S^2

It also provides helpers for normalized scoring against a baseline when needed.

The spatial-balance metric depends on instance-type thresholds. For the current
competition naming convention, we map:
- Clustered_large -> C2
- Clustered_tight -> C1
- Random_large    -> R2
- Random_tight    -> R1
- Mixed_large     -> RC2
- Mixed_tight     -> RC1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import math

from core.constraints import FeasibilityResult, check_solution_feasibility, routes_from_submission_format
from core.parser import VRPTWInstance


@dataclass
class ScoreResult:
    feasible: bool
    objective: float
    total_distance: float
    total_time: float
    vehicles_used: int
    load_variance: float
    spatial_variance: float
    normalized_score: Optional[float] = None
    feasibility: Optional[FeasibilityResult] = None


# Thresholds for 200-customer Solomon-style families.
# These are the values given in the evaluation documentation for size=200.
SPATIAL_THRESHOLDS_200: Dict[str, Tuple[float, float]] = {
    "C1": (44.8, 101.4),
    "C2": (41.6, 91.1),
    "R1": (45.9, 98.0),
    "R2": (45.9, 98.0),
    "RC1": (44.1, 96.6),
    "RC2": (44.1, 96.6),
}


def infer_competition_family(instance_name: str) -> Optional[str]:
    """Infer the benchmark family from the filename.

    Examples
    --------
    Clustered_large_200_4 -> C2
    Mixed_tight_200_29    -> RC1
    Random_tight_200_7    -> R1
    """
    stem = Path(instance_name).stem if "." in instance_name else instance_name
    parts = stem.split("_")
    if len(parts) < 2:
        return None

    dist = parts[0].strip().lower()
    window = parts[1].strip().lower()

    if dist == "clustered":
        return "C1" if window == "tight" else "C2"
    if dist == "random":
        return "R1" if window == "tight" else "R2"
    if dist == "mixed":
        return "RC1" if window == "tight" else "RC2"
    return None


def get_spatial_thresholds(instance_name: str) -> Optional[Tuple[float, float]]:
    family = infer_competition_family(instance_name)
    if family is None:
        return None
    return SPATIAL_THRESHOLDS_200.get(family)


def compute_distance_matrix(instance: VRPTWInstance) -> List[List[float]]:
    nodes = instance.all_nodes
    n = len(nodes)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        xi, yi = nodes[i].x, nodes[i].y
        for j in range(n):
            xj, yj = nodes[j].x, nodes[j].y
            dx = xi - xj
            dy = yi - yj
            matrix[i][j] = math.sqrt(dx * dx + dy * dy)
    return matrix


def _distance(matrix: List[List[float]], i: int, j: int) -> float:
    return matrix[i][j]


def compute_route_distance(route: Sequence[int], distance_matrix: List[List[float]]) -> float:
    dist = 0.0
    for i in range(len(route) - 1):
        dist += _distance(distance_matrix, route[i], route[i + 1])
    return dist


def compute_total_distance(routes: Sequence[Sequence[int]], distance_matrix: List[List[float]]) -> float:
    return sum(compute_route_distance(route, distance_matrix) for route in routes if len(route) >= 2)


def compute_route_time(route: Sequence[int], instance: VRPTWInstance, distance_matrix: List[List[float]]) -> float:
    """Return the final completion time of a single route.

    Travel time equals Euclidean distance. Early arrivals wait until ready_time.
    Service time is added at each customer visit.
    """
    time = 0.0
    for idx in range(len(route) - 1):
        a = route[idx]
        b = route[idx + 1]
        time += _distance(distance_matrix, a, b)
        if b == 0:
            continue
        node = instance.all_nodes[b]
        if time < node.ready_time:
            time = node.ready_time
        time += node.service_time
    return time


def compute_total_time(routes: Sequence[Sequence[int]], instance: VRPTWInstance, distance_matrix: List[List[float]]) -> float:
    return sum(compute_route_time(route, instance, distance_matrix) for route in routes if len(route) >= 2)


def compute_vehicle_count(routes: Sequence[Sequence[int]]) -> int:
    return sum(1 for route in routes if len(route) >= 2)


def compute_route_load(route: Sequence[int], instance: VRPTWInstance) -> float:
    load = 0.0
    for node_id in route:
        if node_id == 0:
            continue
        load += instance.all_nodes[node_id].demand
    return load


def compute_load_variance(routes: Sequence[Sequence[int]], instance: VRPTWInstance) -> float:
    active_routes = [route for route in routes if len(route) >= 2]
    if not active_routes:
        return 0.0
    loads = [compute_route_load(route, instance) for route in active_routes]
    mean_load = sum(loads) / len(loads)
    return sum((load - mean_load) ** 2 for load in loads) / len(loads)


def _categorize_distance(d: float, t1: float, t2: float) -> str:
    if d <= t1:
        return "C"
    if d <= t2:
        return "M"
    return "D"


def compute_spatial_variance(routes: Sequence[Sequence[int]], instance: VRPTWInstance, distance_matrix: List[List[float]]) -> float:
    """Compute the spatial balance variance defined in the evaluation PDF."""
    thresholds = get_spatial_thresholds(instance.name)
    if thresholds is None:
        # If the family is unknown, fall back to 0 to avoid crashing.
        return 0.0
    t1, t2 = thresholds

    active_routes = [route for route in routes if len(route) >= 2]
    if not active_routes:
        return 0.0

    counts_c: List[int] = []
    counts_m: List[int] = []
    counts_d: List[int] = []

    for route in active_routes:
        c = m = d = 0
        for i in range(len(route) - 1):
            seg = _distance(distance_matrix, route[i], route[i + 1])
            category = _categorize_distance(seg, t1, t2)
            if category == "C":
                c += 1
            elif category == "M":
                m += 1
            else:
                d += 1
        counts_c.append(c)
        counts_m.append(m)
        counts_d.append(d)

    def var(values: List[int]) -> float:
        if not values:
            return 0.0
        mean_v = sum(values) / len(values)
        return sum((x - mean_v) ** 2 for x in values) / len(values)

    return var(counts_c) + var(counts_m) + var(counts_d)


def compute_objective(
    routes: Sequence[Sequence[int]],
    instance: VRPTWInstance,
    distance_matrix: List[List[float]],
) -> Tuple[float, Dict[str, float]]:
    """Compute the weighted objective for a feasible solution.

    Returns
    -------
    objective:
        Weighted score J (lower is better).
    components:
        Dictionary with all metrics.
    """
    total_distance = compute_total_distance(routes, distance_matrix)
    total_time = compute_total_time(routes, instance, distance_matrix)
    vehicles_used = compute_vehicle_count(routes)
    load_variance = compute_load_variance(routes, instance)
    spatial_variance = compute_spatial_variance(routes, instance, distance_matrix)

    objective = (
        0.35 * total_time
        + 0.25 * total_distance
        + 0.20 * (vehicles_used * 100.0)
        + 0.12 * load_variance
        + 0.08 * spatial_variance
    )

    components = {
        "total_distance": total_distance,
        "total_time": total_time,
        "vehicles_used": float(vehicles_used),
        "load_variance": load_variance,
        "spatial_variance": spatial_variance,
        "objective": objective,
    }
    return objective, components


def evaluate_solution(
    routes: Sequence[Sequence[int]] | Dict,
    instance: VRPTWInstance,
    distance_matrix: List[List[float]],
    baseline_objective: Optional[float] = None,
) -> ScoreResult:
    """Validate and score a candidate solution.

    Parameters
    ----------
    routes:
        Either a list of routes or a submission-like payload.
    baseline_objective:
        If provided, a normalized score is computed as objective / baseline_objective.
    """
    if isinstance(routes, dict):
        if "solution" in routes and "routes" in routes["solution"]:
            routes = routes["solution"]["routes"]
        elif "routes" in routes:
            routes = routes["routes"]
        else:
            raise ValueError("Unsupported routes dictionary format.")

    normalized_routes = routes_from_submission_format(routes)
    feasibility = check_solution_feasibility(normalized_routes, instance, distance_matrix)

    if not feasibility.feasible:
        return ScoreResult(
            feasible=False,
            objective=float("inf"),
            total_distance=feasibility.total_distance,
            total_time=feasibility.total_time,
            vehicles_used=feasibility.vehicles_used,
            load_variance=float("inf"),
            spatial_variance=float("inf"),
            normalized_score=None,
            feasibility=feasibility,
        )

    objective, components = compute_objective(normalized_routes, instance, distance_matrix)
    normalized = None
    if baseline_objective is not None and baseline_objective > 0:
        normalized = objective / baseline_objective

    return ScoreResult(
        feasible=True,
        objective=objective,
        total_distance=components["total_distance"],
        total_time=components["total_time"],
        vehicles_used=int(components["vehicles_used"]),
        load_variance=components["load_variance"],
        spatial_variance=components["spatial_variance"],
        normalized_score=normalized,
        feasibility=feasibility,
    )


__all__ = [
    "ScoreResult",
    "SPATIAL_THRESHOLDS_200",
    "infer_competition_family",
    "get_spatial_thresholds",
    "compute_distance_matrix",
    "compute_route_distance",
    "compute_total_distance",
    "compute_route_time",
    "compute_total_time",
    "compute_vehicle_count",
    "compute_route_load",
    "compute_load_variance",
    "compute_spatial_variance",
    "compute_objective",
    "evaluate_solution",
]
