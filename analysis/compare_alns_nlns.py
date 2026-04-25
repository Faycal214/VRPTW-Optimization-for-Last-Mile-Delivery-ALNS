from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


FAMILY_ORDER = [
    "Clustered_large",
    "Clustered_tight",
    "Random_large",
    "Random_tight",
    "Mixed_large",
    "Mixed_tight",
]


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


def load_alns(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"instance", "feasible", "total_routes", "load_variance", "spatial_variance", "total_distance", "total_time", "vehicles_used", "objective"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"ALNS CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["total_routes", "load_variance", "spatial_variance", "total_distance", "total_time", "vehicles_used", "objective"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["family"] = df["instance"].apply(parse_family)
    return df


def load_nlns(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "instance" not in df.columns:
        raise ValueError("NLNS CSV missing 'instance' column")

    df = df.copy()
    if "feasible" in df.columns:
        df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["total_routes", "load_variance", "spatial_variance", "total_distance", "total_time", "vehicles_used", "baseline_objective", "final_objective"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["family"] = df["instance"].apply(parse_family)

    if "objective" not in df.columns:
        df["objective"] = df["final_objective"]

    return df


def build_comparison(alns: pd.DataFrame, nlns: pd.DataFrame) -> pd.DataFrame:
    merged = alns.merge(
        nlns,
        on="instance",
        how="inner",
        suffixes=("_alns", "_nlns"),
    )

    merged["objective_gain"] = merged["objective_alns"] - merged["objective_nlns"]
    merged["objective_gain_pct"] = merged["objective_gain"] / merged["objective_alns"].replace(0, pd.NA) * 100.0

    merged["distance_gain"] = merged["total_distance_alns"] - merged["total_distance_nlns"]
    merged["time_gain"] = merged["total_time_alns"] - merged["total_time_nlns"]
    merged["routes_gain"] = merged["total_routes_alns"] - merged["total_routes_nlns"]

    merged["feasible_both"] = merged["feasible_alns"] & merged["feasible_nlns"]
    merged["nlns_better"] = merged["objective_nlns"] < merged["objective_alns"]
    merged["family"] = pd.Categorical(merged["family_alns"], categories=FAMILY_ORDER, ordered=True)

    cols = [
        "instance",
        "family",
        "feasible_alns",
        "feasible_nlns",
        "objective_alns",
        "objective_nlns",
        "objective_gain",
        "objective_gain_pct",
        "total_routes_alns",
        "total_routes_nlns",
        "routes_gain",
        "total_distance_alns",
        "total_distance_nlns",
        "distance_gain",
        "total_time_alns",
        "total_time_nlns",
        "time_gain",
        "vehicles_used_alns",
        "vehicles_used_nlns",
        "feasible_both",
        "nlns_better",
    ]
    return merged[cols].sort_values(["family", "instance"])


def family_summary(comp: pd.DataFrame) -> pd.DataFrame:
    return (
        comp.groupby("family", dropna=False)
        .agg(
            n_instances=("instance", "count"),
            alns_feasible_rate=("feasible_alns", "mean"),
            nlns_feasible_rate=("feasible_nlns", "mean"),
            alns_mean_objective=("objective_alns", "mean"),
            nlns_mean_objective=("objective_nlns", "mean"),
            mean_objective_gain=("objective_gain", "mean"),
            mean_objective_gain_pct=("objective_gain_pct", "mean"),
            alns_mean_routes=("total_routes_alns", "mean"),
            nlns_mean_routes=("total_routes_nlns", "mean"),
            alns_mean_distance=("total_distance_alns", "mean"),
            nlns_mean_distance=("total_distance_nlns", "mean"),
            alns_mean_time=("total_time_alns", "mean"),
            nlns_mean_time=("total_time_nlns", "mean"),
        )
        .reset_index()
        .sort_values("family")
    )


def global_summary(comp: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_instances": len(comp),
                "alns_feasible_rate": comp["feasible_alns"].mean(),
                "nlns_feasible_rate": comp["feasible_nlns"].mean(),
                "alns_mean_objective": comp["objective_alns"].mean(),
                "nlns_mean_objective": comp["objective_nlns"].mean(),
                "mean_objective_gain": comp["objective_gain"].mean(),
                "mean_objective_gain_pct": comp["objective_gain_pct"].mean(),
                "alns_mean_routes": comp["total_routes_alns"].mean(),
                "nlns_mean_routes": comp["total_routes_nlns"].mean(),
                "alns_mean_distance": comp["total_distance_alns"].mean(),
                "nlns_mean_distance": comp["total_distance_nlns"].mean(),
                "alns_mean_time": comp["total_time_alns"].mean(),
                "nlns_mean_time": comp["total_time_nlns"].mean(),
                "nlns_better_rate": comp["nlns_better"].mean(),
            }
        ]
    )


def build_insights(global_df: pd.DataFrame, family_df: pd.DataFrame) -> List[str]:
    g = global_df.iloc[0]
    insights = [
        f"NLNS better rate: {g['nlns_better_rate']:.2%}.",
        f"Mean objective gain (ALNS - NLNS): {g['mean_objective_gain']:.2f} ({g['mean_objective_gain_pct']:.2f}%).",
        f"ALNS feasibility rate: {g['alns_feasible_rate']:.2%}, NLNS feasibility rate: {g['nlns_feasible_rate']:.2%}.",
    ]
    if not family_df.empty:
        best_family = family_df.sort_values("mean_objective_gain_pct", ascending=False).iloc[0]
        worst_family = family_df.sort_values("mean_objective_gain_pct", ascending=True).iloc[0]
        insights.append(
            f"Best family for NLNS gain: {best_family['family']} with {best_family['mean_objective_gain_pct']:.2f}%."
        )
        insights.append(
            f"Weakest family for NLNS gain: {worst_family['family']} with {worst_family['mean_objective_gain_pct']:.2f}%."
        )
    return insights


def write_markdown(
    output_path: Path,
    global_df: pd.DataFrame,
    family_df: pd.DataFrame,
    comp: pd.DataFrame,
    insights: List[str],
) -> None:
    lines = []
    lines.append("# ALNS vs NLNS comparison on test set")
    lines.append("")
    lines.append("## Key insights")
    for item in insights:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Global summary")
    lines.append(global_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Family summary")
    lines.append(family_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Instance-level comparison")
    lines.append(comp.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ALNS and NLNS on the test set.")
    parser.add_argument("--alns_csv", required=True, help="outputs/test/test_summary.csv")
    parser.add_argument("--nlns_csv", required=True, help="outputs/nlns_eval/nlns_summary.csv")
    parser.add_argument("--output_dir", default="analysis_outputs/alns_vs_nlns")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alns = load_alns(args.alns_csv)
    nlns = load_nlns(args.nlns_csv)

    comp = build_comparison(alns, nlns)
    gsum = global_summary(comp)
    fsum = family_summary(comp)
    insights = build_insights(gsum, fsum)

    comp.to_csv(out_dir / "instance_comparison.csv", index=False)
    gsum.to_csv(out_dir / "global_summary.csv", index=False)
    fsum.to_csv(out_dir / "family_summary.csv", index=False)

    report_path = out_dir / "alns_vs_nlns_report.md"
    write_markdown(report_path, gsum, fsum, comp, insights)

    print(f"Saved report to: {report_path}")
    print(f"Saved tables to: {out_dir}")


if __name__ == "__main__":
    main()