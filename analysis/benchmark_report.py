from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

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


def load_alns(csv_path: str, prefix: str = "alns") -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    required = {
        "instance",
        "feasible",
        "total_routes",
        "total_distance",
        "total_time",
        "vehicles_used",
        "objective",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["total_routes", "total_distance", "total_time", "vehicles_used", "objective"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "load_variance" in df.columns:
        df["load_variance"] = pd.to_numeric(df["load_variance"], errors="coerce")
    if "spatial_variance" in df.columns:
        df["spatial_variance"] = pd.to_numeric(df["spatial_variance"], errors="coerce")

    df = df.rename(columns={
        "feasible": f"feasible_{prefix}",
        "total_routes": f"total_routes_{prefix}",
        "total_distance": f"total_distance_{prefix}",
        "total_time": f"total_time_{prefix}",
        "vehicles_used": f"vehicles_used_{prefix}",
        "objective": f"objective_{prefix}",
        "load_variance": f"load_variance_{prefix}",
        "spatial_variance": f"spatial_variance_{prefix}",
    })
    df["family"] = df["instance"].apply(parse_family)
    return df


def load_nlns(csv_path: str, prefix: str = "nlns") -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if "instance" not in df.columns:
        raise ValueError(f"{csv_path} missing instance column")

    if "feasible" in df.columns:
        df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df.rename(columns={"feasible": f"feasible_{prefix}"})
    else:
        df[f"feasible_{prefix}"] = True

    if "objective" in df.columns:
        df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
        df = df.rename(columns={"objective": f"objective_{prefix}"})
    elif "final_objective" in df.columns:
        df["final_objective"] = pd.to_numeric(df["final_objective"], errors="coerce")
        df = df.rename(columns={"final_objective": f"objective_{prefix}"})
    else:
        raise ValueError(f"{csv_path} missing objective/final_objective column")

    for col in ["total_routes", "total_distance", "total_time", "vehicles_used", "load_variance", "spatial_variance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.rename(columns={col: f"{col}_{prefix}"})
        else:
            df[f"{col}_{prefix}"] = pd.NA

    df["family"] = df["instance"].apply(parse_family)
    return df


def load_bks(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if not {"instance", "bks_objective"}.issubset(df.columns):
        raise ValueError("BKS file must have columns: instance, bks_objective")
    df["bks_objective"] = pd.to_numeric(df["bks_objective"], errors="coerce")
    return df[["instance", "bks_objective"]]


def merge_methods(alns: pd.DataFrame, nlns: pd.DataFrame, hybrid: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    comp = alns.merge(nlns, on=["instance", "family"], how="inner")

    comp["objective_gain_nlns_vs_alns"] = comp["objective_alns"] - comp["objective_nlns"]
    comp["objective_gain_pct_nlns_vs_alns"] = (
        comp["objective_gain_nlns_vs_alns"] / comp["objective_alns"].replace(0, pd.NA) * 100.0
    )
    comp["nlns_better"] = comp["objective_nlns"] < comp["objective_alns"]
    comp["tie"] = comp["objective_nlns"] == comp["objective_alns"]

    if hybrid is not None:
        comp = comp.merge(hybrid, on=["instance", "family"], how="left")
        if "objective_hybrid" in comp.columns:
            comp["objective_gain_hybrid_vs_alns"] = comp["objective_alns"] - comp["objective_hybrid"]
            comp["objective_gain_pct_hybrid_vs_alns"] = (
                comp["objective_gain_hybrid_vs_alns"] / comp["objective_alns"].replace(0, pd.NA) * 100.0
            )

    return comp


def add_bks_gap(comp: pd.DataFrame, bks: Optional[pd.DataFrame]) -> pd.DataFrame:
    if bks is None:
        return comp
    out = comp.merge(bks, on="instance", how="left")
    for method in ["alns", "nlns", "hybrid"]:
        col = f"objective_{method}"
        if col in out.columns:
            out[f"gap_to_bks_{method}"] = (out[col] - out["bks_objective"]) / out["bks_objective"].replace(0, pd.NA) * 100.0
    return out


def global_summary(comp: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "method_pair": "NLNS vs ALNS",
            "n_instances": len(comp),
            "alns_feasible_rate": comp["feasible_alns"].mean(),
            "nlns_feasible_rate": comp["feasible_nlns"].mean(),
            "alns_mean_objective": comp["objective_alns"].mean(),
            "nlns_mean_objective": comp["objective_nlns"].mean(),
            "mean_objective_gain": comp["objective_gain_nlns_vs_alns"].mean(),
            "mean_objective_gain_pct": comp["objective_gain_pct_nlns_vs_alns"].mean(),
            "nlns_win_rate": comp["nlns_better"].mean(),
            "tie_rate": comp["tie"].mean(),
        }
    ]
    if "objective_hybrid" in comp.columns:
        rows.append(
            {
                "method_pair": "Hybrid vs ALNS",
                "n_instances": len(comp),
                "alns_feasible_rate": comp["feasible_alns"].mean(),
                "nlns_feasible_rate": pd.NA,
                "alns_mean_objective": comp["objective_alns"].mean(),
                "nlns_mean_objective": comp["objective_hybrid"].mean(),
                "mean_objective_gain": comp["objective_gain_hybrid_vs_alns"].mean(),
                "mean_objective_gain_pct": comp["objective_gain_pct_hybrid_vs_alns"].mean(),
                "nlns_win_rate": (comp["objective_hybrid"] < comp["objective_alns"]).mean(),
                "tie_rate": (comp["objective_hybrid"] == comp["objective_alns"]).mean(),
            }
        )
    return pd.DataFrame(rows)


def family_summary(comp: pd.DataFrame) -> pd.DataFrame:
    g = comp.groupby("family", dropna=False).agg(
        n_instances=("instance", "count"),
        alns_feasible_rate=("feasible_alns", "mean"),
        nlns_feasible_rate=("feasible_nlns", "mean"),
        alns_mean_objective=("objective_alns", "mean"),
        nlns_mean_objective=("objective_nlns", "mean"),
        mean_objective_gain=("objective_gain_nlns_vs_alns", "mean"),
        mean_objective_gain_pct=("objective_gain_pct_nlns_vs_alns", "mean"),
        nlns_win_rate=("nlns_better", "mean"),
        tie_rate=("tie", "mean"),
        alns_mean_routes=("total_routes_alns", "mean"),
        nlns_mean_routes=("total_routes_nlns", "mean"),
        alns_mean_distance=("total_distance_alns", "mean"),
        nlns_mean_distance=("total_distance_nlns", "mean"),
        alns_mean_time=("total_time_alns", "mean"),
        nlns_mean_time=("total_time_nlns", "mean"),
    ).reset_index()

    g["family"] = pd.Categorical(g["family"], categories=FAMILY_ORDER, ordered=True)
    return g.sort_values("family")


def instance_table(comp: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "instance",
        "family",
        "feasible_alns",
        "feasible_nlns",
        "objective_alns",
        "objective_nlns",
        "objective_gain_nlns_vs_alns",
        "objective_gain_pct_nlns_vs_alns",
        "total_routes_alns",
        "total_routes_nlns",
        "total_distance_alns",
        "total_distance_nlns",
        "total_time_alns",
        "total_time_nlns",
        "vehicles_used_alns",
        "vehicles_used_nlns",
        "nlns_better",
        "tie",
    ]
    if "objective_hybrid" in comp.columns:
        cols += ["objective_hybrid", "objective_gain_hybrid_vs_alns", "objective_gain_pct_hybrid_vs_alns"]
    if "bks_objective" in comp.columns:
        cols += ["bks_objective"]
        for method in ["alns", "nlns", "hybrid"]:
            col = f"gap_to_bks_{method}"
            if col in comp.columns:
                cols.append(col)
    cols = [c for c in cols if c in comp.columns]
    return comp[cols].sort_values(["family", "instance"])


def build_insights(global_df: pd.DataFrame, family_df: pd.DataFrame) -> List[str]:
    g = global_df.iloc[0]
    insights = [
        f"NLNS win rate over ALNS: {g['nlns_win_rate']:.2%}.",
        f"Mean objective gain (ALNS - NLNS): {g['mean_objective_gain']:.2f} ({g['mean_objective_gain_pct']:.2f}%).",
    ]
    if len(global_df) > 1 and "method_pair" in global_df.columns:
        h = global_df.iloc[1]
        insights.append(
            f"Hybrid mean objective gain over ALNS: {h['mean_objective_gain']:.2f} ({h['mean_objective_gain_pct']:.2f}%)."
        )
    if not family_df.empty:
        best = family_df.sort_values("mean_objective_gain_pct", ascending=False).iloc[0]
        worst = family_df.sort_values("mean_objective_gain_pct", ascending=True).iloc[0]
        insights.append(f"Best family for NLNS gain: {best['family']} ({best['mean_objective_gain_pct']:.2f}%).")
        insights.append(f"Weakest family for NLNS gain: {worst['family']} ({worst['mean_objective_gain_pct']:.2f}%).")
    return insights


def write_report(output_path: Path, global_df: pd.DataFrame, family_df: pd.DataFrame, inst_df: pd.DataFrame, insights: List[str]) -> None:
    lines: List[str] = []
    lines.append("# Benchmark report")
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
    lines.append("## Instance comparison")
    lines.append(inst_df.to_markdown(index=False, floatfmt=".4f"))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark study for ALNS / NLNS / hybrid.")
    parser.add_argument("--alns_csv", required=True)
    parser.add_argument("--nlns_csv", required=True)
    parser.add_argument("--hybrid_csv", default=None)
    parser.add_argument("--bks_csv", default=None)
    parser.add_argument("--output_dir", default="analysis_outputs/benchmark")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alns = load_alns(args.alns_csv, prefix="alns")
    nlns = load_nlns(args.nlns_csv, prefix="nlns")
    hybrid = load_nlns(args.hybrid_csv, prefix="hybrid") if args.hybrid_csv else None
    bks = load_bks(args.bks_csv) if args.bks_csv else None

    comp = merge_methods(alns, nlns, hybrid=hybrid)
    comp = add_bks_gap(comp, bks)

    gsum = global_summary(comp)
    fsum = family_summary(comp)
    inst = instance_table(comp)
    insights = build_insights(gsum, fsum)

    comp.to_csv(out_dir / "comparison.csv", index=False)
    gsum.to_csv(out_dir / "global_summary.csv", index=False)
    fsum.to_csv(out_dir / "family_summary.csv", index=False)
    inst.to_csv(out_dir / "instance_comparison.csv", index=False)
    write_report(out_dir / "benchmark_report.md", gsum, fsum, inst, insights)

    print(f"Saved report to: {out_dir / 'benchmark_report.md'}")
    print(f"Saved tables to: {out_dir}")


if __name__ == "__main__":
    main()