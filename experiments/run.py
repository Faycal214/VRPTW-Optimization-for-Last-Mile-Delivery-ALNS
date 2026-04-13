"""Experiment runner for VRPTW ALNS.

This version:
- runs the baseline / ALNS pipeline
- writes one JSON file per instance
- writes one flattened summary CSV per split (train/test)
- keeps train/test output folders separated automatically

JSON files keep only the compact structure requested:
- instance
- feasible
- constraints
- evaluation
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from core.parser import parse_instances_dir, VRPTWInstance
from core.baseline import build_baseline_solution
from core.constraints import check_solution_feasibility
from core.evaluation import compute_distance_matrix, compute_objective
from core.model import Solution, Route
from alns.alns import run_alns


# -----------------------------------------------------------------------------
# CONFIG + SUMMARY
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    instances_dir: str
    output_dir: str = "outputs"
    alns_iterations: int = 100
    seed: int = 42
    use_baseline_only: bool = False


@dataclass
class ExperimentSummary:
    rows: List[Dict]
    feasible_count: int
    mean_objective: float
    mean_total_distance: float
    mean_total_time: float
    summary_csv_path: str
    output_dir: str


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------

def compute_load_variance(routes: List[List[int]], instance: VRPTWInstance) -> float:
    loads = [sum(instance.all_nodes[n].demand for n in route if n != 0) for route in routes]
    if not loads:
        return 0.0
    mean = sum(loads) / len(loads)
    return sum((l - mean) ** 2 for l in loads) / len(loads)


def compute_spatial_variance(routes: List[List[int]], instance: VRPTWInstance) -> float:
    centroids = []

    for route in routes:
        coords = [(instance.all_nodes[n].x, instance.all_nodes[n].y) for n in route if n != 0]
        if not coords:
            continue

        cx = sum(x for x, _ in coords) / len(coords)
        cy = sum(y for _, y in coords) / len(coords)
        centroids.append((cx, cy))

    if not centroids:
        return 0.0

    mx = sum(x for x, _ in centroids) / len(centroids)
    my = sum(y for _, y in centroids) / len(centroids)

    return sum((x - mx) ** 2 + (y - my) ** 2 for x, y in centroids) / len(centroids)


def _coverage_report(routes: List[List[int]], num_customers: int) -> Tuple[int, List[int], List[int]]:
    """Return unique visited count, missing customers, and duplicates."""
    visited = set()
    duplicates = set()

    for route in routes:
        for node in route:
            if node == 0:
                continue
            if node in visited:
                duplicates.add(node)
            visited.add(node)

    expected = set(range(1, num_customers + 1))
    missing = sorted(expected - visited)
    duplicates = sorted(duplicates)
    return len(visited), missing, duplicates


# -----------------------------------------------------------------------------
# JSON OUTPUT
# -----------------------------------------------------------------------------

def build_json_output(
    instance_name: str,
    feasible: bool,
    total_routes: int,
    total_distance: float,
    total_time: float,
    vehicles_used: int,
    load_variance: float,
    spatial_variance: float,
    objective: float,
    errors: list,
    missing_customers: list,
    duplicate_customers: list,
    route_details: list | None = None,
) -> dict:
    return {
        "instance": instance_name,
        "feasible": feasible,
        "constraints": {
            "total_routes": total_routes,
            "total_distance": total_distance,
            "total_time": total_time,
            "vehicles_used": vehicles_used,
            "route_details": route_details or [],
            "errors": errors or [],
            "missing_customers": missing_customers or [],
            "duplicate_customers": duplicate_customers or [],
        },
        "evaluation": {
            "load_variance": load_variance,
            "spatial_variance": spatial_variance,
            "total_distance": total_distance,
            "total_time": total_time,
            "vehicles_used": vehicles_used,
            "objective": objective,
        },
    }


def write_instance_json(
    instance: VRPTWInstance,
    routes: List[List[int]],
    feasibility,
    objective: float,
    components: Dict,
    output_path: Path,
) -> None:
    route_details = [list(route) for route in routes]
    _, missing_customers, duplicate_customers = _coverage_report(routes, instance.num_nodes - 1)

    payload = build_json_output(
        instance_name=instance.name,
        feasible=bool(getattr(feasibility, "feasible", False)),
        total_routes=len(routes),
        total_distance=float(getattr(feasibility, "total_distance", components.get("total_distance", 0.0))),
        total_time=float(getattr(feasibility, "total_time", components.get("total_time", 0.0))),
        vehicles_used=int(getattr(feasibility, "vehicles_used", components.get("vehicles_used", len(routes)))),
        load_variance=float(components.get("load_variance", 0.0)),
        spatial_variance=float(components.get("spatial_variance", 0.0)),
        objective=float(objective),
        errors=list(getattr(feasibility, "errors", [])),
        missing_customers=missing_customers,
        duplicate_customers=duplicate_customers,
        route_details=route_details,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------------------------------------------------------------
# CSV SUMMARY
# -----------------------------------------------------------------------------

def _json_list(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_summary_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "instance",
        "feasible",
        "total_routes",
        "num_customers",
        "errors",
        "missing_customers",
        "duplicate_customers",
        "load_variance",
        "spatial_variance",
        "total_distance",
        "total_time",
        "vehicles_used",
        "objective",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for row in rows:
            routes = row["routes"]
            num_customers = len({n for route in routes for n in route if n != 0})
            feasibility = row["feasibility"]
            components = row["components"]
            _, missing_customers, duplicate_customers = _coverage_report(routes, num_customers)

            writer.writerow({
                "instance": row["instance_name"],
                "feasible": row["feasible"],
                "total_routes": len(routes),
                "num_customers": num_customers,
                "errors": _json_list(list(getattr(feasibility, "errors", []))),
                "missing_customers": _json_list(missing_customers),
                "duplicate_customers": _json_list(duplicate_customers),
                "load_variance": components.get("load_variance", 0.0),
                "spatial_variance": components.get("spatial_variance", 0.0),
                "total_distance": components.get("total_distance", 0.0),
                "total_time": components.get("total_time", 0.0),
                "vehicles_used": components.get("vehicles_used", len(routes)),
                "objective": row["objective"],
            })


# -----------------------------------------------------------------------------
# SINGLE INSTANCE
# -----------------------------------------------------------------------------

def run_single_instance(instance: VRPTWInstance, config: ExperimentConfig) -> Dict:
    print(f"Starting instance: {instance.name}", flush=True)

    distance_matrix = compute_distance_matrix(instance)

    print("Building baseline solution...", flush=True)
    baseline = build_baseline_solution(instance)

    solution = Solution(
        routes=[Route(path=r) for r in baseline.routes],
        unserved_customers=[],
    )

    if config.use_baseline_only:
        print("Skipping ALNS (baseline only mode)", flush=True)
    else:
        print(f"Running ALNS ({config.alns_iterations} iterations)...", flush=True)
        solution = run_alns(
            instance=instance,
            initial_solution=solution,
            distance_matrix=distance_matrix,
            max_iterations=config.alns_iterations,
            seed=config.seed,
        )

    routes = solution.route_paths
    feasibility = check_solution_feasibility(routes, instance, distance_matrix)

    if feasibility.feasible:
        objective, components = compute_objective(routes, instance, distance_matrix)
    else:
        objective = float("inf")
        components = {
            "total_distance": feasibility.total_distance,
            "total_time": feasibility.total_time,
            "vehicles_used": feasibility.vehicles_used,
            "load_variance": compute_load_variance(routes, instance),
            "spatial_variance": compute_spatial_variance(routes, instance),
        }

    return {
        "instance_name": instance.name,
        "routes": routes,
        "feasible": bool(feasibility.feasible),
        "feasibility": feasibility,
        "objective": objective,
        "components": components,
        "instance": instance,
    }


# -----------------------------------------------------------------------------
# MAIN EXPERIMENT LOOP
# -----------------------------------------------------------------------------

def run_experiments(config: ExperimentConfig) -> ExperimentSummary:
    print("=== VRPTW RUN STARTED ===", flush=True)

    instances = parse_instances_dir(config.instances_dir)
    print(f"Loaded {len(instances)} instances.", flush=True)

    split_name = Path(config.instances_dir).name.lower().strip() or "instances"
    base_output_dir = Path(config.output_dir) / split_name
    json_output_dir = base_output_dir / "instances_json"
    json_output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    for idx, instance in enumerate(instances, 1):
        print(f"[{idx}/{len(instances)}] Processing {instance.name}", flush=True)

        result = run_single_instance(instance, config)

        output_path = json_output_dir / f"{instance.name}.json"
        write_instance_json(
            instance=result["instance"],
            routes=result["routes"],
            feasibility=result["feasibility"],
            objective=result["objective"],
            components=result["components"],
            output_path=output_path,
        )

        print(
            f"Finished {instance.name} | "
            f"feasible={result['feasible']} | "
            f"objective={result['objective']} | "
            f"routes={len(result['routes'])}",
            flush=True,
        )

        results.append(result)

    feasible_count = sum(1 for r in results if r["feasible"])
    valid_objectives = [r["objective"] for r in results if r["objective"] != float("inf")]

    mean_objective = sum(valid_objectives) / max(1, len(valid_objectives))
    mean_total_distance = sum(r["components"]["total_distance"] for r in results) / max(1, len(results))
    mean_total_time = sum(r["components"]["total_time"] for r in results) / max(1, len(results))

    summary_csv_path = base_output_dir / f"{split_name}_summary.csv"
    write_summary_csv(results, summary_csv_path)

    print(f"Saved summary CSV: {summary_csv_path}", flush=True)
    print("=== RUN COMPLETED ===", flush=True)

    return ExperimentSummary(
        rows=results,
        feasible_count=feasible_count,
        mean_objective=mean_objective,
        mean_total_distance=mean_total_distance,
        mean_total_time=mean_total_time,
        summary_csv_path=str(summary_csv_path),
        output_dir=str(base_output_dir),
    )


__all__ = [
    "ExperimentConfig",
    "ExperimentSummary",
    "compute_load_variance",
    "compute_spatial_variance",
    "build_json_output",
    "write_summary_csv",
    "write_instance_json",
    "run_single_instance",
    "run_experiments",
]
