from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

from alns.destroy import destroy_solution
from alns.repair import repair_solution
from core.baseline import build_baseline_solution
from core.constraints import check_solution_feasibility
from core.evaluation import compute_distance_matrix, compute_objective
from core.model import Route, Solution
from core.parser import parse_instances_dir, VRPTWInstance

from nlns.policy import ACTION_SPACE, OperatorPolicyNet


def route_load(route: List[int], instance: VRPTWInstance) -> float:
    return sum(instance.all_nodes[n].demand for n in route if n != 0)


def route_distance(route: List[int], instance: VRPTWInstance) -> float:
    dist = 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        dx = instance.all_nodes[a].x - instance.all_nodes[b].x
        dy = instance.all_nodes[a].y - instance.all_nodes[b].y
        dist += (dx * dx + dy * dy) ** 0.5
    return dist


def compute_load_variance(routes: List[List[int]], instance: VRPTWInstance) -> float:
    loads = [route_load(route, instance) for route in routes if len(route) > 2]
    if not loads:
        return 0.0
    mean = sum(loads) / len(loads)
    return sum((x - mean) ** 2 for x in loads) / len(loads)


def compute_spatial_variance(routes: List[List[int]], instance: VRPTWInstance) -> float:
    centroids = []
    for route in routes:
        pts = [(instance.all_nodes[n].x, instance.all_nodes[n].y) for n in route if n != 0]
        if not pts:
            continue
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        centroids.append((cx, cy))

    if not centroids:
        return 0.0

    mx = sum(x for x, _ in centroids) / len(centroids)
    my = sum(y for _, y in centroids) / len(centroids)
    return sum((x - mx) ** 2 + (y - my) ** 2 for x, y in centroids) / len(centroids)


def solution_from_routes(routes: List[List[int]]) -> Solution:
    sol = Solution()
    for route in routes:
        sol.routes.append(Route(path=list(route)))
    sol.unserved_customers = []
    return sol


def extract_state(
    instance: VRPTWInstance,
    solution: Solution,
    distance_matrix: List[List[float]],
    step: int,
    max_steps: int,
    baseline_objective: float,
) -> torch.Tensor:
    routes = solution.route_paths
    feas = check_solution_feasibility(routes, instance, distance_matrix)

    if feas.feasible:
        objective, components = compute_objective(routes, instance, distance_matrix)
        total_distance = components["total_distance"]
        total_time = components["total_time"]
    else:
        objective = baseline_objective * 2.0
        total_distance = feas.total_distance
        total_time = feas.total_time

    num_customers = max(1, instance.num_nodes - 1)
    served = set(solution.all_customers())
    unserved = num_customers - len(served)

    routes_used = len(solution.active_routes())
    avg_route_len = sum(max(0, len(r.path) - 2) for r in solution.active_routes()) / max(1, routes_used)
    avg_load = sum(route_load(r.path, instance) for r in solution.active_routes()) / max(1, routes_used)

    state = torch.tensor(
        [
            step / max(1, max_steps),
            len(routes) / max(1, instance.num_vehicles),
            routes_used / max(1, instance.num_vehicles),
            unserved / max(1, num_customers),
            objective / max(1.0, baseline_objective),
            total_distance / max(1.0, baseline_objective),
            total_time / max(1.0, baseline_objective),
            compute_load_variance(routes, instance) / max(1.0, instance.capacity ** 2),
            compute_spatial_variance(routes, instance) / 10000.0,
            avg_route_len / max(1.0, num_customers),
            avg_load / max(1.0, instance.capacity),
            1.0 if feas.feasible else 0.0,
        ],
        dtype=torch.float32,
    )
    return state


def collect_coverage(routes: List[List[int]], instance: VRPTWInstance) -> Tuple[List[int], List[int]]:
    visited: List[int] = []
    duplicates: List[int] = []
    seen: Set[int] = set()

    for route in routes:
        for node in route:
            if node == 0:
                continue
            if node in seen and node not in duplicates:
                duplicates.append(node)
            seen.add(node)
            visited.append(node)

    expected = set(range(1, instance.num_nodes))
    missing = sorted(list(expected - seen))
    return missing, sorted(duplicates)


def build_json_output(
    instance: VRPTWInstance,
    routes: List[List[int]],
    feasibility,
    objective: float,
    components: Dict,
) -> Dict:
    missing_customers, duplicate_customers = collect_coverage(routes, instance)

    route_details = []
    for route in routes:
        route_details.append(
            {
                "path": route,
                "num_customers": len([n for n in route if n != 0]),
                "distance": route_distance(route, instance),
                "load": route_load(route, instance),
            }
        )

    return {
        "instance": instance.name,
        "feasible": bool(feasibility.feasible),
        "constraints": {
            "total_routes": len(routes),
            "total_distance": float(getattr(feasibility, "total_distance", components.get("total_distance", 0.0))),
            "total_time": float(getattr(feasibility, "total_time", components.get("total_time", 0.0))),
            "vehicles_used": int(getattr(feasibility, "vehicles_used", components.get("vehicles_used", len(routes)))),
            "route_details": route_details,
            "errors": list(getattr(feasibility, "errors", [])),
            "missing_customers": missing_customers,
            "duplicate_customers": duplicate_customers,
        },
        "evaluation": {
            "load_variance": float(components.get("load_variance", 0.0)),
            "spatial_variance": float(components.get("spatial_variance", 0.0)),
            "total_distance": float(components.get("total_distance", 0.0)),
            "total_time": float(components.get("total_time", 0.0)),
            "vehicles_used": int(components.get("vehicles_used", len(routes))),
            "objective": float(objective),
        },
    }


def load_policy(model_path: str) -> OperatorPolicyNet:
    checkpoint = torch.load(model_path, map_location="cpu")
    input_dim = int(checkpoint.get("input_dim", 12))
    hidden_dim = int(checkpoint.get("hidden_dim", 128))

    model = OperatorPolicyNet(input_dim=input_dim, hidden_dim=hidden_dim, n_actions=len(ACTION_SPACE))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def choose_action(model: OperatorPolicyNet, state: torch.Tensor) -> int:
    with torch.no_grad():
        logits, _ = model(state)
        action = int(torch.argmax(logits).item())
    return action


def solve_instance(
    instance: VRPTWInstance,
    model: OperatorPolicyNet,
    steps_per_instance: int,
    min_remove: int,
    max_remove: int,
    seed: int,
) -> Dict:
    rng = random.Random(seed)
    distance_matrix = compute_distance_matrix(instance)

    baseline = build_baseline_solution(instance)
    solution = solution_from_routes(baseline.routes)

    if not baseline.feasible:
        feasibility = check_solution_feasibility(solution.route_paths, instance, distance_matrix)
        objective = float("inf")
        components = {
            "total_distance": feasibility.total_distance,
            "total_time": feasibility.total_time,
            "vehicles_used": feasibility.vehicles_used,
            "load_variance": compute_load_variance(solution.route_paths, instance),
            "spatial_variance": compute_spatial_variance(solution.route_paths, instance),
        }
        return {
            "instance": instance.name,
            "routes": solution.route_paths,
            "feasibility": feasibility,
            "objective": objective,
            "components": components,
            "baseline_objective": baseline.objective,
        }

    current_obj = baseline.objective

    for step in range(steps_per_instance):
        state = extract_state(
            instance=instance,
            solution=solution,
            distance_matrix=distance_matrix,
            step=step,
            max_steps=steps_per_instance,
            baseline_objective=baseline.objective,
        )

        action_idx = choose_action(model, state)
        destroy_name, repair_name = ACTION_SPACE[action_idx]

        num_remove = rng.randint(min_remove, max_remove)

        destroy_result = destroy_solution(
            solution=solution,
            instance=instance,
            num_remove=num_remove,
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
        candidate_routes = candidate.route_paths
        feas = check_solution_feasibility(candidate_routes, instance, distance_matrix)

        if feas.feasible:
            cand_obj, _ = compute_objective(candidate_routes, instance, distance_matrix)
            if cand_obj < current_obj:
                solution = candidate.copy()
                current_obj = cand_obj

    final_routes = solution.route_paths
    feasibility = check_solution_feasibility(final_routes, instance, distance_matrix)

    if feasibility.feasible:
        objective, components = compute_objective(final_routes, instance, distance_matrix)
    else:
        objective = float("inf")
        components = {
            "total_distance": feasibility.total_distance,
            "total_time": feasibility.total_time,
            "vehicles_used": feasibility.vehicles_used,
            "load_variance": compute_load_variance(final_routes, instance),
            "spatial_variance": compute_spatial_variance(final_routes, instance),
        }

    return {
        "instance": instance.name,
        "routes": final_routes,
        "feasibility": feasibility,
        "objective": objective,
        "components": components,
        "baseline_objective": baseline.objective,
    }


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
        "baseline_objective",
        "final_objective",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for row in rows:
            routes = row["routes"]
            feasibility = row["feasibility"]
            components = row["components"]
            num_customers = len({n for route in routes for n in route if n != 0})

            writer.writerow(
                {
                    "instance": row["instance"],
                    "feasible": int(feasibility.feasible),
                    "total_routes": len(routes),
                    "num_customers": num_customers,
                    "errors": json.dumps(list(getattr(feasibility, "errors", [])), ensure_ascii=False),
                    "missing_customers": json.dumps([], ensure_ascii=False),
                    "duplicate_customers": json.dumps([], ensure_ascii=False),
                    "load_variance": components.get("load_variance", 0.0),
                    "spatial_variance": components.get("spatial_variance", 0.0),
                    "total_distance": components.get("total_distance", 0.0),
                    "total_time": components.get("total_time", 0.0),
                    "vehicles_used": components.get("vehicles_used", len(routes)),
                    "baseline_objective": row["baseline_objective"],
                    "final_objective": row["objective"],
                }
            )


def run_inference(
    instances_dir: str,
    model_path: str,
    output_dir: str,
    steps_per_instance: int = 25,
    min_remove: int = 5,
    max_remove: int = 20,
    seed: int = 42,
) -> None:
    instances = parse_instances_dir(instances_dir)
    if not instances:
        raise FileNotFoundError(f"No instances found in {instances_dir}")

    model = load_policy(model_path)

    out_root = Path(output_dir)
    json_dir = out_root / "instances_json"
    json_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    print("=== NLNS INFERENCE STARTED ===", flush=True)
    print(f"Loaded {len(instances)} instances.", flush=True)

    for idx, instance in enumerate(instances, 1):
        print(f"[{idx}/{len(instances)}] Solving {instance.name}", flush=True)

        result = solve_instance(
            instance=instance,
            model=model,
            steps_per_instance=steps_per_instance,
            min_remove=min_remove,
            max_remove=max_remove,
            seed=seed + idx,
        )
        results.append(result)

        feasibility = result["feasibility"]
        components = result["components"]

        payload = build_json_output(
            instance=instance,
            routes=result["routes"],
            feasibility=feasibility,
            objective=result["objective"],
            components=components,
        )

        output_path = json_dir / f"{instance.name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"Finished {instance.name} | feasible={feasibility.feasible} | "
            f"objective={result['objective']} | routes={len(result['routes'])}",
            flush=True,
        )

    summary_csv_path = out_root / "nlns_summary.csv"
    write_summary_csv(results, summary_csv_path)

    feasible_count = sum(1 for r in results if r["feasibility"].feasible)
    valid_objectives = [r["objective"] for r in results if r["objective"] != float("inf")]
    mean_objective = sum(valid_objectives) / max(1, len(valid_objectives))
    mean_total_distance = sum(r["components"]["total_distance"] for r in results) / max(1, len(results))
    mean_total_time = sum(r["components"]["total_time"] for r in results) / max(1, len(results))

    print("=== NLNS INFERENCE COMPLETED ===", flush=True)
    print(f"Feasible instances: {feasible_count}/{len(results)}", flush=True)
    print(f"Mean objective: {mean_objective:.4f}", flush=True)
    print(f"Mean total distance: {mean_total_distance:.4f}", flush=True)
    print(f"Mean total time: {mean_total_time:.4f}", flush=True)
    print(f"Summary CSV: {summary_csv_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained NLNS policy on the test instances and export JSON outputs.")
    parser.add_argument("--instances_dir", required=True, help="e.g. data/test")
    parser.add_argument("--model_path", default="outputs/nlns/checkpoints/final_model.pt")
    parser.add_argument("--output_dir", default="outputs/nlns_eval")
    parser.add_argument("--steps_per_instance", type=int, default=25)
    parser.add_argument("--min_remove", type=int, default=5)
    parser.add_argument("--max_remove", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_inference(
        instances_dir=args.instances_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        steps_per_instance=args.steps_per_instance,
        min_remove=args.min_remove,
        max_remove=args.max_remove,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()