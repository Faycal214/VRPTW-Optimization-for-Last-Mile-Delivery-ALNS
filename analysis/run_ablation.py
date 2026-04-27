from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alns.alns import run_alns
from alns.destroy import destroy_solution
from alns.repair import repair_solution
from core.baseline import build_baseline_solution
from core.constraints import check_solution_feasibility
from core.evaluation import compute_distance_matrix, compute_objective
from core.model import Route, Solution
from core.parser import parse_instances_dir
from nlns.policy import ACTION_SPACE, OperatorPolicyNet


DEFAULT_CONFIGS = [
    {"name": "ALNS", "mode": "alns"},
    {"name": "NLNS", "mode": "nlns"},
    {"name": "Hybrid_default", "mode": "hybrid", "destroy": 0.3, "iters": 100},
    {"name": "Hybrid_low_destroy", "mode": "hybrid", "destroy": 0.1, "iters": 100},
    {"name": "Hybrid_high_destroy", "mode": "hybrid", "destroy": 0.5, "iters": 100},
    {"name": "Hybrid_few_steps", "mode": "hybrid", "steps": 10},
    {"name": "Hybrid_many_steps", "mode": "hybrid", "steps": 50},
]

DEFAULT_DESTROY_RATIO = 0.3
DEFAULT_ALNS_ITERS = 100
DEFAULT_RL_STEPS = 25


def slugify(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def infer_family(instance_name: str) -> str:
    parts = str(instance_name).split("_")
    if len(parts) < 2:
        return "Unknown"
    family = f"{parts[0].capitalize()}_{parts[1].lower()}"
    if family in {"Clustered_large", "Clustered_tight", "Random_large", "Random_tight", "Mixed_large", "Mixed_tight"}:
        return family
    return "Unknown"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def solution_from_routes(routes: List[List[int]]) -> Solution:
    route_objs = [Route(path=list(route)) for route in routes]
    try:
        return Solution(routes=route_objs, unserved_customers=[])
    except TypeError:
        sol = Solution()
        sol.routes = route_objs
        if hasattr(sol, "unserved_customers"):
            sol.unserved_customers = []
        return sol


def routes_from_solution(solution) -> List[List[int]]:
    if hasattr(solution, "route_paths"):
        return [list(r) for r in solution.route_paths]
    if hasattr(solution, "routes"):
        routes: List[List[int]] = []
        for r in solution.routes:
            if isinstance(r, list):
                routes.append(list(r))
            elif hasattr(r, "path"):
                routes.append(list(r.path))
            else:
                routes.append(list(r))
        return routes
    raise TypeError(f"Unsupported solution object: {type(solution)!r}")


def route_load(route: List[int], instance) -> float:
    return sum(instance.all_nodes[n].demand for n in route if n != 0)


def route_distance(route: List[int], instance) -> float:
    dist = 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        dx = instance.all_nodes[a].x - instance.all_nodes[b].x
        dy = instance.all_nodes[a].y - instance.all_nodes[b].y
        dist += (dx * dx + dy * dy) ** 0.5
    return dist


def compute_load_variance(routes: List[List[int]], instance) -> float:
    loads = [route_load(route, instance) for route in routes if len(route) > 2]
    if not loads:
        return 0.0
    mean = sum(loads) / len(loads)
    return sum((x - mean) ** 2 for x in loads) / len(loads)


def compute_spatial_variance(routes: List[List[int]], instance) -> float:
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


def evaluate_routes(instance, routes: List[List[int]], distance_matrix) -> Tuple[bool, float, Dict, object]:
    feasibility = check_solution_feasibility(routes, instance, distance_matrix)

    if feasibility.feasible:
        objective, components = compute_objective(routes, instance, distance_matrix)
    else:
        objective = float("inf")
        components = {
            "total_distance": getattr(feasibility, "total_distance", 0.0),
            "total_time": getattr(feasibility, "total_time", 0.0),
            "vehicles_used": getattr(feasibility, "vehicles_used", len(routes)),
            "load_variance": compute_load_variance(routes, instance),
            "spatial_variance": compute_spatial_variance(routes, instance),
        }

    components.setdefault("total_distance", 0.0)
    components.setdefault("total_time", 0.0)
    components.setdefault("vehicles_used", len(routes))
    components.setdefault("load_variance", 0.0)
    components.setdefault("spatial_variance", 0.0)

    return bool(feasibility.feasible), float(objective), components, feasibility


def load_policy(model_path: str) -> OperatorPolicyNet:
    checkpoint = torch.load(model_path, map_location="cpu")
    input_dim = int(checkpoint.get("input_dim", 12))
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    model = OperatorPolicyNet(input_dim=input_dim, hidden_dim=hidden_dim, n_actions=len(ACTION_SPACE))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def choose_action(model: OperatorPolicyNet, state) -> int:
    with torch.no_grad():
        logits, _ = model(state)
        return int(torch.argmax(logits).item())


def extract_state(
    instance,
    solution: Solution,
    distance_matrix,
    step: int,
    max_steps: int,
    baseline_objective: float,
) -> "torch.Tensor":
    import torch

    routes = routes_from_solution(solution)
    feas, objective, components, _ = evaluate_routes(instance, routes, distance_matrix)

    if not feas:
        objective = baseline_objective * 2.0
        total_distance = components["total_distance"]
        total_time = components["total_time"]
    else:
        total_distance = components["total_distance"]
        total_time = components["total_time"]

    num_customers = max(1, instance.num_nodes - 1)
    served = {n for route in routes for n in route if n != 0}
    unserved = num_customers - len(served)

    active_routes = [r for r in routes if len(r) > 2]
    routes_used = len(active_routes)
    avg_route_len = (
        sum(max(0, len(r) - 2) for r in active_routes) / max(1, routes_used)
    )
    avg_load = sum(route_load(r, instance) for r in active_routes) / max(1, routes_used)

    return torch.tensor(
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
            1.0 if feas else 0.0,
        ],
        dtype=torch.float32,
    )


def run_nlns_rollout(
    instance,
    policy: OperatorPolicyNet,
    start_routes: List[List[int]],
    steps: int,
    destroy_ratio: float,
    seed: int,
    baseline_objective: float,
):
    import torch

    rng = random.Random(seed)
    distance_matrix = compute_distance_matrix(instance)

    current_routes = [list(r) for r in start_routes]
    current_solution = solution_from_routes(current_routes)

    feas, objective, _, _ = evaluate_routes(instance, current_routes, distance_matrix)
    current_obj = objective if np.isfinite(objective) else 1e18

    num_customers = max(1, instance.num_nodes - 1)
    num_remove = max(1, min(num_customers - 1, int(round(destroy_ratio * num_customers))))

    for step in range(steps):
        state = extract_state(
            instance=instance,
            solution=current_solution,
            distance_matrix=distance_matrix,
            step=step,
            max_steps=steps,
            baseline_objective=baseline_objective if np.isfinite(baseline_objective) else current_obj,
        )

        action_idx = choose_action(policy, state)
        destroy_name, repair_name = ACTION_SPACE[action_idx]

        destroy_result = destroy_solution(
            solution=current_solution,
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
        candidate_routes = routes_from_solution(candidate)
        cand_feas, cand_obj, _, _ = evaluate_routes(instance, candidate_routes, distance_matrix)

        if cand_feas and cand_obj < current_obj:
            current_routes = [list(r) for r in candidate_routes]
            current_solution = solution_from_routes(current_routes)
            current_obj = cand_obj

    final_feas, final_obj, final_components, final_feas_obj = evaluate_routes(instance, current_routes, distance_matrix)
    return {
        "routes": current_routes,
        "feasible": final_feas,
        "objective": final_obj,
        "components": final_components,
        "feasibility": final_feas_obj,
    }


def run_alns_baseline(instance, initial_routes: List[List[int]], iters: int, seed: int):
    distance_matrix = compute_distance_matrix(instance)
    initial_solution = solution_from_routes(initial_routes)
    result = run_alns(
        instance=instance,
        initial_solution=initial_solution,
        distance_matrix=distance_matrix,
        max_iterations=iters,
        seed=seed,
    )
    final_routes = routes_from_solution(result)
    feasible, objective, components, feasibility = evaluate_routes(instance, final_routes, distance_matrix)
    return {
        "routes": final_routes,
        "feasible": feasible,
        "objective": objective,
        "components": components,
        "feasibility": feasibility,
    }


def write_instance_json(
    path: Path,
    instance,
    config: Dict,
    routes: List[List[int]],
    feasible: bool,
    objective: float,
    components: Dict,
    feasibility,
    runtime_sec: float,
    extra: Dict | None = None,
):
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

    payload = {
        "instance": instance.name,
        "family": infer_family(instance.name),
        "config": config,
        "feasible": bool(feasible),
        "constraints": {
            "total_routes": len(routes),
            "total_distance": float(components.get("total_distance", 0.0)),
            "total_time": float(components.get("total_time", 0.0)),
            "vehicles_used": int(components.get("vehicles_used", len(routes))),
            "route_details": route_details,
            "errors": list(getattr(feasibility, "errors", [])),
            "missing_customers": sorted(list(getattr(feasibility, "missing_customers", []))),
            "duplicate_customers": sorted(list(getattr(feasibility, "duplicate_customers", []))),
        },
        "evaluation": {
            "load_variance": float(components.get("load_variance", 0.0)),
            "spatial_variance": float(components.get("spatial_variance", 0.0)),
            "total_distance": float(components.get("total_distance", 0.0)),
            "total_time": float(components.get("total_time", 0.0)),
            "vehicles_used": int(components.get("vehicles_used", len(routes))),
            "objective": float(objective),
            "runtime_sec": float(runtime_sec),
        },
    }

    if extra:
        payload["evaluation"].update(extra)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_results(df: pd.DataFrame, alns_df: pd.DataFrame, config: Dict) -> Dict:
    merged = df.merge(alns_df, on=["instance", "family"], suffixes=("_cfg", "_alns"))
    merged["objective_cfg"] = pd.to_numeric(merged["objective_cfg"], errors="coerce")
    merged["objective_alns"] = pd.to_numeric(merged["objective_alns"], errors="coerce")

    valid = np.isfinite(merged["objective_cfg"]) & np.isfinite(merged["objective_alns"])
    gains = merged.loc[valid, "objective_alns"] - merged.loc[valid, "objective_cfg"]
    gain_pcts = gains / merged.loc[valid, "objective_alns"].replace(0, np.nan) * 100.0

    summary = {
        "config": config["name"],
        "mode": config["mode"],
        "destroy": float(config.get("destroy", DEFAULT_DESTROY_RATIO)),
        "iters": int(config.get("iters", DEFAULT_ALNS_ITERS)),
        "steps": int(config.get("steps", DEFAULT_RL_STEPS)),
        "n_instances": int(len(df)),
        "feasible_rate": float(df["feasible"].mean()),
        "mean_objective": float(df.loc[np.isfinite(df["objective"]), "objective"].mean()),
        "mean_total_distance": float(df["total_distance"].mean()),
        "mean_total_time": float(df["total_time"].mean()),
        "mean_vehicles": float(df["vehicles_used"].mean()),
        "mean_routes": float(df["total_routes"].mean()),
        "mean_runtime_sec": float(df["runtime_sec"].mean()),
        "mean_gain_vs_alns": float(gains.mean()) if len(gains) else 0.0,
        "mean_gain_pct_vs_alns": float(gain_pcts.mean()) if len(gain_pcts) else 0.0,
        "win_rate_vs_alns": float((merged.loc[valid, "objective_cfg"] < merged.loc[valid, "objective_alns"]).mean()) if len(merged.loc[valid]) else 0.0,
        "tie_rate_vs_alns": float((merged.loc[valid, "objective_cfg"] == merged.loc[valid, "objective_alns"]).mean()) if len(merged.loc[valid]) else 0.0,
    }
    return summary


def expand_config(cfg: Dict) -> Dict:
    out = {
        "destroy": DEFAULT_DESTROY_RATIO,
        "iters": DEFAULT_ALNS_ITERS,
        "steps": DEFAULT_RL_STEPS,
    }
    out.update(cfg)
    return out


def run_config(instance_list, policy, cfg: Dict, output_root: Path, seed: int) -> pd.DataFrame:
    cfg = expand_config(cfg)
    config_name = cfg["name"]
    config_dir = ensure_dir(output_root / slugify(config_name))
    json_dir = ensure_dir(config_dir / "instances_json")

    rows = []
    print(f"\n=== Running {config_name} ===", flush=True)

    for idx, instance in enumerate(instance_list, 1):
        print(f"[{config_name}] {idx}/{len(instance_list)} -> {instance.name}", flush=True)
        t0 = time.perf_counter()

        baseline = build_baseline_solution(instance)
        baseline_routes = routes_from_solution(baseline)
        distance_matrix = compute_distance_matrix(instance)
        baseline_feasible, baseline_objective, _, baseline_feas = evaluate_routes(instance, baseline_routes, distance_matrix)

        try:
            if cfg["mode"] == "alns":
                result = run_alns_baseline(
                    instance=instance,
                    initial_routes=baseline_routes,
                    iters=int(cfg["iters"]),
                    seed=seed + idx,
                )
                final_routes = result["routes"]
                final_feasible = result["feasible"]
                final_objective = result["objective"]
                final_components = result["components"]
                final_feas = result["feasibility"]
                extra = {"baseline_objective": baseline_objective}

            elif cfg["mode"] == "nlns":
                result = run_nlns_rollout(
                    instance=instance,
                    policy=policy,
                    start_routes=baseline_routes,
                    steps=int(cfg["steps"]),
                    destroy_ratio=float(cfg["destroy"]),
                    seed=seed + idx,
                    baseline_objective=baseline_objective,
                )
                final_routes = result["routes"]
                final_feasible = result["feasible"]
                final_objective = result["objective"]
                final_components = result["components"]
                final_feas = result["feasibility"]
                extra = {
                    "baseline_objective": baseline_objective,
                    "rl_stage_objective": final_objective,
                }

            elif cfg["mode"] == "hybrid":
                rl_stage = run_nlns_rollout(
                    instance=instance,
                    policy=policy,
                    start_routes=baseline_routes,
                    steps=int(cfg["steps"]),
                    destroy_ratio=float(cfg["destroy"]),
                    seed=seed + idx,
                    baseline_objective=baseline_objective,
                )

                rl_solution = solution_from_routes(rl_stage["routes"])
                alns_solution = run_alns(
                    instance=instance,
                    initial_solution=rl_solution,
                    distance_matrix=distance_matrix,
                    max_iterations=int(cfg["iters"]),
                    seed=seed + idx,
                )
                final_routes = routes_from_solution(alns_solution)
                final_feasible, final_objective, final_components, final_feas = evaluate_routes(instance, final_routes, distance_matrix)
                extra = {
                    "baseline_objective": baseline_objective,
                    "rl_stage_objective": rl_stage["objective"],
                }
            else:
                raise ValueError(f"Unknown mode: {cfg['mode']}")

        except Exception as exc:
            print(f"[{config_name}] {instance.name} failed: {exc}", flush=True)
            final_routes = baseline_routes
            final_feasible = baseline_feasible
            final_objective = baseline_objective
            final_components = {
                "total_distance": 0.0,
                "total_time": 0.0,
                "vehicles_used": len(final_routes),
                "load_variance": compute_load_variance(final_routes, instance),
                "spatial_variance": compute_spatial_variance(final_routes, instance),
            }
            final_feas = baseline_feas
            extra = {"error": str(exc), "baseline_objective": baseline_objective}

        elapsed = time.perf_counter() - t0

        row = {
            "instance": instance.name,
            "family": infer_family(instance.name),
            "feasible": bool(final_feasible),
            "total_routes": len(final_routes),
            "num_customers": len({n for route in final_routes for n in route if n != 0}),
            "total_distance": float(final_components.get("total_distance", 0.0)),
            "total_time": float(final_components.get("total_time", 0.0)),
            "vehicles_used": int(final_components.get("vehicles_used", len(final_routes))),
            "objective": float(final_objective),
            "runtime_sec": float(elapsed),
        }
        rows.append(row)

        write_instance_json(
            path=json_dir / f"{instance.name}.json",
            instance=instance,
            config=cfg,
            routes=final_routes,
            feasible=final_feasible,
            objective=final_objective,
            components=final_components,
            feasibility=final_feas,
            runtime_sec=elapsed,
            extra=extra,
        )

        print(
            f"[{config_name}] {instance.name}: feasible={final_feasible} "
            f"objective={final_objective:.4f} routes={len(final_routes)}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(config_dir / "summary.csv", index=False)
    return df


def build_wide_comparison(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = results["ALNS"][["instance", "family", "objective", "feasible", "total_routes", "total_distance", "total_time", "vehicles_used"]].copy()
    base = base.rename(
        columns={
            "objective": "objective_ALNS",
            "feasible": "feasible_ALNS",
            "total_routes": "total_routes_ALNS",
            "total_distance": "total_distance_ALNS",
            "total_time": "total_time_ALNS",
            "vehicles_used": "vehicles_used_ALNS",
        }
    )

    for name, df in results.items():
        if name == "ALNS":
            continue
        tmp = df[["instance", "family", "objective", "feasible", "total_routes", "total_distance", "total_time", "vehicles_used"]].copy()
        tmp = tmp.rename(
            columns={
                "objective": f"objective_{name}",
                "feasible": f"feasible_{name}",
                "total_routes": f"total_routes_{name}",
                "total_distance": f"total_distance_{name}",
                "total_time": f"total_time_{name}",
                "vehicles_used": f"vehicles_used_{name}",
            }
        )
        base = base.merge(tmp, on=["instance", "family"], how="inner")
        base[f"gain_vs_ALNS_{name}"] = base["objective_ALNS"] - base[f"objective_{name}"]
        base[f"gain_pct_vs_ALNS_{name}"] = (
            base[f"gain_vs_ALNS_{name}"] / base["objective_ALNS"].replace(0, np.nan) * 100.0
        )
        base[f"better_vs_ALNS_{name}"] = base[f"objective_{name}"] < base["objective_ALNS"]

    return base.sort_values(["family", "instance"])


def build_report(summary_df: pd.DataFrame, comparison_df: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append("# Ablation report")
    lines.append("")
    lines.append("## Config summary")
    lines.append(summary_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Instance comparison")
    lines.append(comparison_df.to_markdown(index=False, floatfmt=".4f"))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_ablation(
    instances_dir: str,
    model_path: str,
    output_dir: str,
    seed: int,
    configs: List[Dict],
) -> None:
    ensure_dir(output_dir)
    instance_list = parse_instances_dir(instances_dir)
    if not instance_list:
        raise FileNotFoundError(f"No instances found in {instances_dir}")

    policy = load_policy(model_path)

    results: Dict[str, pd.DataFrame] = {}
    summaries: List[Dict] = []

    for cfg in configs:
        cfg = expand_config(cfg)
        df = run_config(instance_list, policy, cfg, Path(output_dir), seed=seed)
        results[cfg["name"]] = df

    alns_df = results["ALNS"]
    for cfg in configs:
        cfg = expand_config(cfg)
        summary = summarize_results(results[cfg["name"]], alns_df, cfg)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(Path(output_dir) / "ablation_summary.csv", index=False)

    comparison_df = build_wide_comparison(results)
    comparison_df.to_csv(Path(output_dir) / "instance_comparison.csv", index=False)

    build_report(summary_df, comparison_df, Path(output_dir) / "ablation_report.md")

    print("\n=== Ablation completed ===", flush=True)
    print(summary_df.to_markdown(index=False, floatfmt=".4f"), flush=True)
    print(f"Saved summary to: {Path(output_dir) / 'ablation_summary.csv'}", flush=True)
    print(f"Saved comparison to: {Path(output_dir) / 'instance_comparison.csv'}", flush=True)
    print(f"Saved report to: {Path(output_dir) / 'ablation_report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation runner for ALNS / NLNS / Hybrid on VRPTW.")
    parser.add_argument("--instances_dir", required=True, help="e.g. data/test")
    parser.add_argument("--model_path", default="outputs/nlns/checkpoints/final_model.pt")
    parser.add_argument("--output_dir", default="outputs/ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_ablation(
        instances_dir=args.instances_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        seed=args.seed,
        configs=DEFAULT_CONFIGS,
    )


if __name__ == "__main__":
    import torch  # local import so the module can load even if torch is absent in some environments

    main()