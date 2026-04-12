"""Experiment runner, plotting helpers, and submission export for VRPTW.

This module orchestrates the full evaluation pipeline:
- load instances
- build an initial solution
- optionally run ALNS
- score the solution
- save results
- export Kaggle submission CSV
- plot routes for inspection

It depends on the project modules created earlier:
- core.parser
- core.model
- core.constraints
- core.evaluation
- core.baseline
- alns.*
- utils.plot_routes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import csv
import json
import random

from core.baseline import build_baseline_solution
from core.constraints import check_solution_feasibility
from core.evaluation import compute_distance_matrix, compute_objective
from core.model import Route, Solution
from core.parser import VRPTWInstance, parse_instances_dir
from alns.alns import run_alns
from utils.plot_routes import plot_routes


@dataclass
class ExperimentConfig:
    instances_dir: str
    output_dir: str = "outputs"
    seed: int = 42
    save_plots: bool = True
    save_results_csv: bool = True
    save_submission_csv: bool = True
    use_baseline_only: bool = False
    alns_iterations: int = 1000


@dataclass
class InstanceRunResult:
    instance_name: str
    feasible: bool
    objective: float
    total_distance: float
    total_time: float
    vehicles_used: int
    load_variance: float
    spatial_variance: float
    notes: List[str]
    routes: List[List[int]]


@dataclass
class ExperimentSummary:
    rows: List[InstanceRunResult]
    mean_objective: float
    mean_total_distance: float
    mean_total_time: float
    feasible_count: int



def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
# Solution conversion helpers
# -----------------------------------------------------------------------------

def solution_to_routes(solution: Solution) -> List[List[int]]:
    """Convert a Solution object to plain route paths for scoring/submission."""
    return [route.path for route in solution.active_routes()]


def baseline_solution_as_solution_object(instance: VRPTWInstance) -> Solution:
    """Build baseline and convert it into the common Solution dataclass."""
    base = build_baseline_solution(instance)

    sol = Solution()
    for route_path in base.routes:
        sol.routes.append(Route(path=list(route_path)))

    sol.feasible = base.feasible
    sol.objective = base.objective
    sol.total_distance = base.total_distance
    sol.total_time = base.total_time
    sol.vehicles_used = base.vehicles_used
    sol.unserved_customers = []
    return sol


# -----------------------------------------------------------------------------
# Submission export
# -----------------------------------------------------------------------------

def write_submission_csv(rows: Sequence[Tuple[str, Sequence[Sequence[int]]]], output_path: str) -> None:
    """Write Kaggle-format submission CSV."""
    output = Path(output_path)
    _ensure_dir(output.parent)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "routes"])
        for instance_id, routes in rows:
            writer.writerow([instance_id, json.dumps([list(route) for route in routes])])


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_single_instance(
    instance: VRPTWInstance,
    config: ExperimentConfig,
    rng: Optional[random.Random] = None,
) -> InstanceRunResult:
    """Run the configured solver on one instance."""
    rng = rng or random.Random(config.seed)
    distance_matrix = compute_distance_matrix(instance)

    base_solution = baseline_solution_as_solution_object(instance)

    if config.use_baseline_only:
        solution = base_solution
    else:
        solution = run_alns(
            instance=instance,
            initial_solution=base_solution,
            distance_matrix=distance_matrix,
            max_iterations=config.alns_iterations,
            seed=config.seed,
        )

    routes = solution_to_routes(solution)
    feasibility = check_solution_feasibility(routes, instance, distance_matrix)

    if feasibility.feasible:
        objective, components = compute_objective(routes, instance, distance_matrix)
    else:
        objective = float("inf")
        components = {
            "total_distance": feasibility.total_distance,
            "total_time": feasibility.total_time,
            "vehicles_used": float(feasibility.vehicles_used),
            "load_variance": float("inf"),
            "spatial_variance": float("inf"),
        }

    if config.save_plots:
        plot_path = Path(config.output_dir) / "plots" / f"{instance.name}.png"
        plot_routes(instance, routes, output_path=str(plot_path), show=False)

    return InstanceRunResult(
        instance_name=instance.name,
        feasible=feasibility.feasible,
        objective=objective,
        total_distance=components["total_distance"],
        total_time=components["total_time"],
        vehicles_used=int(components["vehicles_used"]),
        load_variance=components["load_variance"],
        spatial_variance=components["spatial_variance"],
        notes=feasibility.errors,
        routes=routes,
    )


def run_experiments(config: ExperimentConfig) -> ExperimentSummary:
    """Run the solver on all instances in a directory."""
    _ensure_dir(config.output_dir)
    instances = parse_instances_dir(config.instances_dir)

    rows: List[InstanceRunResult] = []
    for instance in instances:
        result = run_single_instance(instance, config=config, rng=random.Random(config.seed))
        rows.append(result)

    feasible_count = sum(1 for r in rows if r.feasible)
    valid_objectives = [r.objective for r in rows if r.objective != float("inf")]
    mean_objective = sum(valid_objectives) / max(1, len(valid_objectives))
    mean_total_distance = sum(r.total_distance for r in rows) / max(1, len(rows))
    mean_total_time = sum(r.total_time for r in rows) / max(1, len(rows))

    if config.save_results_csv:
        results_csv = Path(config.output_dir) / "results.csv"
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "instance_name",
                    "feasible",
                    "objective",
                    "total_distance",
                    "total_time",
                    "vehicles_used",
                    "load_variance",
                    "spatial_variance",
                    "notes",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r.instance_name,
                        int(r.feasible),
                        r.objective,
                        r.total_distance,
                        r.total_time,
                        r.vehicles_used,
                        r.load_variance,
                        r.spatial_variance,
                        " | ".join(r.notes),
                    ]
                )

    if config.save_submission_csv:
        submission_rows = [(r.instance_name, r.routes) for r in rows]
        write_submission_csv(submission_rows, str(Path(config.output_dir) / "submission.csv"))

    return ExperimentSummary(
        rows=rows,
        mean_objective=mean_objective,
        mean_total_distance=mean_total_distance,
        mean_total_time=mean_total_time,
        feasible_count=feasible_count,
    )


# -----------------------------------------------------------------------------
# CLI-style entry point
# -----------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run VRPTW experiments.")
    parser.add_argument("--instances_dir", type=str, required=True, help="Directory containing .TXT instances")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to write results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_plots", action="store_true", help="Disable route plots")
    parser.add_argument("--no_results_csv", action="store_true", help="Do not write results.csv")
    parser.add_argument("--no_submission_csv", action="store_true", help="Do not write submission.csv")
    parser.add_argument("--baseline_only", action="store_true", help="Run only the baseline heuristic")
    parser.add_argument("--alns_iterations", type=int, default=1000, help="Number of ALNS iterations")

    args = parser.parse_args()

    config = ExperimentConfig(
        instances_dir=args.instances_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        save_plots=not args.no_plots,
        save_results_csv=not args.no_results_csv,
        save_submission_csv=not args.no_submission_csv,
        use_baseline_only=args.baseline_only,
        alns_iterations=args.alns_iterations,
    )

    summary = run_experiments(config)
    print(f"Feasible instances: {summary.feasible_count}/{len(summary.rows)}")
    print(f"Mean objective: {summary.mean_objective:.4f}")
    print(f"Mean total distance: {summary.mean_total_distance:.4f}")
    print(f"Mean total time: {summary.mean_total_time:.4f}")


if __name__ == "__main__":
    main()


__all__ = [
    "ExperimentConfig",
    "InstanceRunResult",
    "ExperimentSummary",
    "solution_to_routes",
    "baseline_solution_as_solution_object",
    "write_submission_csv",
    "run_single_instance",
    "run_experiments",
    "main",
]
