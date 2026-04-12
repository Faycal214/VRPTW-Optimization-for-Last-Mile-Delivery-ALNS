"""Project entry point for the VRPTW solver."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.run import ExperimentConfig, run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="VRPTW solver entry point")

    parser.add_argument("--instances_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no_results_csv", action="store_true")
    parser.add_argument("--no_submission_csv", action="store_true")
    parser.add_argument("--no_instance_json", action="store_true")

    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--alns_iterations", type=int, default=30)

    args = parser.parse_args()

    instances_path = Path(args.instances_dir)
    if not instances_path.exists():
        raise FileNotFoundError(f"Instances directory not found: {instances_path}")

    instance_count = len(list(instances_path.glob("*.TXT")))

    print("=== VRPTW RUN STARTED ===", flush=True)
    print(f"Instances directory: {instances_path}", flush=True)
    print(f"Instances found: {instance_count}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(
        f"Mode: {'baseline only' if args.baseline_only else f'ALNS ({args.alns_iterations} iterations)'}",
        flush=True,
    )
    print("==========================", flush=True)

    config = ExperimentConfig(
        instances_dir=args.instances_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        alns_iterations=args.alns_iterations,
    )

    summary = run_experiments(config)

    print("=== VRPTW RUN SUMMARY ===", flush=True)
    print(f"Feasible instances: {summary.feasible_count}/{len(summary.rows)}", flush=True)
    print(f"Mean objective: {summary.mean_objective:.4f}", flush=True)
    print(f"Mean total distance: {summary.mean_total_distance:.4f}", flush=True)
    print(f"Mean total time: {summary.mean_total_time:.4f}", flush=True)
    print("=========================", flush=True)


if __name__ == "__main__":
    main()