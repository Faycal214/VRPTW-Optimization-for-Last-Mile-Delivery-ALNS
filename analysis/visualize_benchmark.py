from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.parser import parse_instance


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17a2b8",
    "#6610f2",
]

FAMILY_ORDER = [
    "Clustered_large",
    "Clustered_tight",
    "Random_large",
    "Random_tight",
    "Mixed_large",
    "Mixed_tight",
]


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def parse_family(instance_name: str) -> str:
    parts = str(instance_name).split("_")
    if len(parts) < 2:
        return "Unknown"
    distribution = parts[0].strip().capitalize()
    window = parts[1].strip().lower()
    if distribution not in {"Clustered", "Random", "Mixed"}:
        return "Unknown"
    if window not in {"large", "tight"}:
        return "Unknown"
    return f"{distribution}_{window}"


def find_file_by_stem(root_dir: str | Path, stem: str, suffix: str | None = None) -> Path:
    root = Path(root_dir)
    matches = sorted(root.rglob(f"{stem}.*"))
    if suffix is not None:
        matches = [p for p in matches if p.suffix.lower() == suffix.lower()]
    if not matches:
        raise FileNotFoundError(f"Could not find file for stem '{stem}' under {root}")
    return matches[0]


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_routes(payload: dict) -> List[List[int]]:
    routes: List[List[int]] = []
    constraints = payload.get("constraints", {})
    route_details = constraints.get("route_details", [])

    if not route_details and "routes" in payload:
        route_details = payload["routes"]

    for item in route_details:
        if isinstance(item, dict):
            path = item.get("path") or item.get("route") or item.get("nodes")
            if path is None:
                continue
            routes.append([int(x) for x in path])
        elif isinstance(item, list):
            routes.append([int(x) for x in item])

    return routes


def route_distance(route: List[int], instance) -> float:
    dist = 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        dx = instance.all_nodes[a].x - instance.all_nodes[b].x
        dy = instance.all_nodes[a].y - instance.all_nodes[b].y
        dist += (dx * dx + dy * dy) ** 0.5
    return dist


def route_load(route: List[int], instance) -> float:
    return sum(instance.all_nodes[n].demand for n in route if n != 0)


def route_midpoint(route: List[int], instance) -> Tuple[float, float]:
    pts = [(instance.all_nodes[n].x, instance.all_nodes[n].y) for n in route]
    if not pts:
        return 0.0, 0.0
    mid = len(pts) // 2
    return pts[mid]


def route_label_summary(routes: List[List[int]]) -> str:
    n_routes = len(routes)
    n_customers = len({n for route in routes for n in route if n != 0})
    return f"Routes: {n_routes} | Customers served: {n_customers}"


def axis_limits(instance, pad_ratio: float = 0.06) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = [n.x for n in instance.all_nodes]
    ys = [n.y for n in instance.all_nodes]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(1.0, (x_max - x_min) * pad_ratio)
    y_pad = max(1.0, (y_max - y_min) * pad_ratio)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


# -----------------------------------------------------------------------------
# Benchmark comparison utilities
# -----------------------------------------------------------------------------

def select_representative_instance(comparison_csv: Path, family: Optional[str] = None) -> str:
    df = pd.read_csv(comparison_csv)
    if "instance" not in df.columns:
        raise ValueError(f"{comparison_csv} missing 'instance' column")

    if "family" not in df.columns:
        df["family"] = df["instance"].apply(parse_family)

    for col in ["total_routes_alns", "total_routes_nlns", "objective_gain_pct_nlns_vs_alns"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "total_routes_alns" in df.columns and "total_routes_nlns" in df.columns:
        df["route_count"] = df[["total_routes_alns", "total_routes_nlns"]].min(axis=1)
    elif "total_routes_alns" in df.columns:
        df["route_count"] = df["total_routes_alns"]
    elif "total_routes_nlns" in df.columns:
        df["route_count"] = df["total_routes_nlns"]
    else:
        df["route_count"] = pd.NA

    if family is not None and family != "auto":
        df = df[df["family"] == family]
        if df.empty:
            raise ValueError(f"No instances found for family {family}")

    if "route_count" in df.columns and df["route_count"].notna().any():
        idx = df["route_count"].astype(float).idxmin()
    elif "objective_gain_pct_nlns_vs_alns" in df.columns:
        idx = df["objective_gain_pct_nlns_vs_alns"].astype(float).idxmax()
    else:
        idx = df.index[0]

    return str(df.loc[idx, "instance"])


# -----------------------------------------------------------------------------
# Small-multiples route plotting
# -----------------------------------------------------------------------------

def plot_small_multiples(
    instance_path: Path,
    routes: List[List[int]],
    title: str,
    output_path: Path,
    dpi: int = 220,
) -> None:
    instance = parse_instance(str(instance_path))
    xlim, ylim = axis_limits(instance)

    n_routes = len(routes)
    if n_routes == 0:
        raise ValueError(f"No routes found for {title}")

    ncols = 2 if n_routes <= 6 else 3
    nrows = math.ceil(n_routes / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 5.5, nrows * 5.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if n_routes == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    else:
        axes = axes.reshape(nrows, ncols)

    all_customers_x = [n.x for n in instance.all_nodes[1:]]
    all_customers_y = [n.y for n in instance.all_nodes[1:]]

    for idx, route in enumerate(routes):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        color = PALETTE[idx % len(PALETTE)]

        # Background customers
        ax.scatter(
            all_customers_x,
            all_customers_y,
            s=18,
            color="#b0b0b0",
            alpha=0.16,
            linewidths=0,
            zorder=1,
        )

        # Highlight only customers used by this route
        route_nodes = [n for n in route if n != 0]
        route_x = [instance.all_nodes[n].x for n in route]
        route_y = [instance.all_nodes[n].y for n in route]
        used_x = [instance.all_nodes[n].x for n in route_nodes]
        used_y = [instance.all_nodes[n].y for n in route_nodes]

        ax.plot(route_x, route_y, color=color, linewidth=2.8, zorder=3)
        ax.scatter(used_x, used_y, s=42, color=color, edgecolors="white", linewidths=0.4, zorder=4)

        # Depot
        ax.scatter(
            [instance.all_nodes[0].x],
            [instance.all_nodes[0].y],
            s=140,
            marker="*",
            color="#f58518",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        ax.text(
            instance.all_nodes[0].x,
            instance.all_nodes[0].y,
            " Depot",
            fontsize=9,
            va="bottom",
            ha="left",
            color="black",
            zorder=6,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_title(
            f"Route {idx + 1}\nDist: {route_distance(route, instance):.1f} | Load: {route_load(route, instance):.0f}",
            fontsize=10,
        )
        ax.tick_params(labelsize=8)

    # hide unused axes
    for idx in range(n_routes, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=15, y=1.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Learning curve plot
# -----------------------------------------------------------------------------

def plot_learning_curve(rewards_csv: Path, output_path: Path, dpi: int = 220) -> None:
    df = pd.read_csv(rewards_csv)
    if "episode_reward" not in df.columns:
        raise ValueError(f"{rewards_csv} missing 'episode_reward' column")

    df = df.copy()
    if "epoch" not in df.columns:
        df["epoch"] = 1
    if "episode" not in df.columns:
        df["episode"] = range(1, len(df) + 1)

    for col in ["episode_reward", "baseline_objective", "final_objective"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["epoch", "episode"]).reset_index(drop=True)
    df["global_step"] = range(1, len(df) + 1)
    df["reward_ma20"] = df["episode_reward"].rolling(window=20, min_periods=1).mean()

    epoch_summary = (
        df.groupby("epoch", as_index=False)
        .agg(
            mean_reward=("episode_reward", "mean"),
            mean_baseline_objective=("baseline_objective", "mean"),
            mean_final_objective=("final_objective", "mean"),
        )
        .sort_values("epoch")
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)

    axes[0].plot(df["global_step"], df["episode_reward"], linewidth=1.2, alpha=0.35, label="Episode reward")
    axes[0].plot(df["global_step"], df["reward_ma20"], linewidth=2.8, label="Reward moving average (20)")
    axes[0].set_title("NLNS training reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(False)

    axes[1].plot(epoch_summary["epoch"], epoch_summary["mean_baseline_objective"], marker="o", linewidth=2, label="Baseline objective")
    axes[1].plot(epoch_summary["epoch"], epoch_summary["mean_final_objective"], marker="o", linewidth=2, label="Final objective")
    axes[1].set_title("Objective by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Objective")
    axes[1].legend()
    axes[1].grid(False)

    fig.suptitle("NLNS training curve", fontsize=15, y=1.02)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Family-level benchmark plot
# -----------------------------------------------------------------------------

def plot_family_gain(comparison_csv: Path, output_path: Path, dpi: int = 220) -> None:
    df = pd.read_csv(comparison_csv)
    if "family" not in df.columns:
        raise ValueError(f"{comparison_csv} missing 'family' column")
    if "objective_gain_pct_nlns_vs_alns" not in df.columns:
        raise ValueError(f"{comparison_csv} missing 'objective_gain_pct_nlns_vs_alns' column")

    df = df.copy()
    df["objective_gain_pct_nlns_vs_alns"] = pd.to_numeric(df["objective_gain_pct_nlns_vs_alns"], errors="coerce")
    df["nlns_better"] = df["objective_nlns"] < df["objective_alns"] if {"objective_nlns", "objective_alns"}.issubset(df.columns) else False

    family = (
        df.groupby("family", as_index=False)
        .agg(
            mean_gain_pct=("objective_gain_pct_nlns_vs_alns", "mean"),
            median_gain_pct=("objective_gain_pct_nlns_vs_alns", "median"),
            win_rate=("nlns_better", "mean"),
            n_instances=("instance", "count"),
        )
        .sort_values("family")
    )
    family["family"] = pd.Categorical(family["family"], categories=FAMILY_ORDER, ordered=True)
    family = family.sort_values("family")

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    bars = ax.bar(family["family"].astype(str), family["mean_gain_pct"], color="#4c78a8", alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Family-level objective gain of NLNS over ALNS")
    ax.set_xlabel("Family")
    ax.set_ylabel("Mean objective gain (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(False)

    for bar, win_rate in zip(bars, family["win_rate"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"win {win_rate:.0%}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Small-multiples route visualization for Solomon VRPTW instances.")
    parser.add_argument("--instances_dir", required=True, help="Original TXT instances directory, e.g. data/test")
    parser.add_argument("--alns_json_dir", required=True, help="ALNS JSON outputs directory")
    parser.add_argument("--nlns_json_dir", required=True, help="NLNS JSON outputs directory")
    parser.add_argument("--rewards_csv", required=True, help="outputs/nlns/logs/episode_rewards.csv")
    parser.add_argument("--comparison_csv", required=True, help="analysis_outputs/benchmark/comparison.csv")
    parser.add_argument("--output_dir", default="analysis_outputs/visuals")
    parser.add_argument("--family", default="Clustered_large", help="Family to target when auto-selecting an instance")
    parser.add_argument("--instance", default="auto", help="Specific instance stem, or 'auto'")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.instance == "auto":
        instance_name = select_representative_instance(Path(args.comparison_csv), args.family)
    else:
        instance_name = args.instance

    instance_txt = find_file_by_stem(args.instances_dir, instance_name)
    alns_json = find_file_by_stem(args.alns_json_dir, instance_name, suffix=".json")
    nlns_json = find_file_by_stem(args.nlns_json_dir, instance_name, suffix=".json")

    alns_payload = load_json(alns_json)
    nlns_payload = load_json(nlns_json)
    alns_routes = extract_routes(alns_payload)
    nlns_routes = extract_routes(nlns_payload)

    plot_small_multiples(
        instance_path=instance_txt,
        routes=alns_routes,
        title=f"ALNS small-multiples — {instance_name}",
        output_path=out_dir / f"alns_small_multiples_{instance_name}.png",
        dpi=args.dpi,
    )
    plot_small_multiples(
        instance_path=instance_txt,
        routes=nlns_routes,
        title=f"NLNS small-multiples — {instance_name}",
        output_path=out_dir / f"nlns_small_multiples_{instance_name}.png",
        dpi=args.dpi,
    )
    plot_learning_curve(Path(args.rewards_csv), out_dir / "training_curve.png", dpi=args.dpi)
    plot_family_gain(Path(args.comparison_csv), out_dir / "family_gain.png", dpi=args.dpi)

    summary_path = out_dir / "visual_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Visual summary",
                "",
                f"Selected family: {args.family}",
                f"Selected instance: {instance_name}",
                "",
                f"ALNS figure: {out_dir / f'alns_small_multiples_{instance_name}.png'}",
                f"NLNS figure: {out_dir / f'nlns_small_multiples_{instance_name}.png'}",
                f"Training curve: {out_dir / 'training_curve.png'}",
                f"Family gain plot: {out_dir / 'family_gain.png'}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved ALNS small-multiples to: {out_dir / f'alns_small_multiples_{instance_name}.png'}")
    print(f"Saved NLNS small-multiples to: {out_dir / f'nlns_small_multiples_{instance_name}.png'}")
    print(f"Saved training curve to: {out_dir / 'training_curve.png'}")
    print(f"Saved family gain plot to: {out_dir / 'family_gain.png'}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
