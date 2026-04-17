from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


FAMILY_ORDER = [
    "Clustered_large",
    "Clustered_tight",
    "Random_large",
    "Random_tight",
    "Mixed_large",
    "Mixed_tight",
]

NUMERIC_COLUMNS = [
    "total_routes",
    "num_customers",
    "load_variance",
    "spatial_variance",
    "total_distance",
    "total_time",
    "vehicles_used",
    "objective",
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


def load_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "instance" not in df.columns:
        raise ValueError(f"Missing 'instance' column in {csv_path}")

    df = df.copy()
    df["family"] = df["instance"].apply(parse_family)

    if "feasible" in df.columns:
        df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1", "yes"])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["objective"] = df["objective"].replace([float("inf"), float("-inf")], pd.NA)
    return df


def aggregate_by_family(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("family", dropna=False)
        .agg(
            n_instances=("instance", "count"),
            feasible_rate=("feasible", "mean"),
            mean_total_routes=("total_routes", "mean"),
            mean_num_customers=("num_customers", "mean"),
            mean_load_variance=("load_variance", "mean"),
            mean_spatial_variance=("spatial_variance", "mean"),
            mean_total_distance=("total_distance", "mean"),
            mean_total_time=("total_time", "mean"),
            mean_vehicles_used=("vehicles_used", "mean"),
            mean_objective=("objective", "mean"),
            median_objective=("objective", "median"),
            min_objective=("objective", "min"),
            max_objective=("objective", "max"),
        )
        .reset_index()
    )
    summary["family"] = pd.Categorical(summary["family"], categories=FAMILY_ORDER, ordered=True)
    return summary.sort_values("family")


def split_overview(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "n_instances": len(df),
            "feasible_rate": df["feasible"].mean(),
            "mean_routes": df["total_routes"].mean(),
            "mean_customers": df["num_customers"].mean(),
            "mean_distance": df["total_distance"].mean(),
            "mean_time": df["total_time"].mean(),
            "mean_objective": df["objective"].mean(),
            "median_objective": df["objective"].median(),
        }
    ])


def distribution_window_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for distribution in ["Clustered", "Random", "Mixed"]:
        large_family = f"{distribution}_large"
        tight_family = f"{distribution}_tight"

        large = df[df["family"] == large_family]
        tight = df[df["family"] == tight_family]

        if large.empty or tight.empty:
            continue

        large_means = large[NUMERIC_COLUMNS].mean(numeric_only=True)
        tight_means = tight[NUMERIC_COLUMNS].mean(numeric_only=True)

        row = {
            "distribution": distribution,
            "large_family": large_family,
            "tight_family": tight_family,
            "large_instances": len(large),
            "tight_instances": len(tight),
        }

        for metric in [
            "total_routes",
            "num_customers",
            "load_variance",
            "spatial_variance",
            "total_distance",
            "total_time",
            "vehicles_used",
            "objective",
        ]:
            a = large_means.get(metric, pd.NA)
            b = tight_means.get(metric, pd.NA)
            row[f"large_{metric}"] = a
            row[f"tight_{metric}"] = b
            row[f"diff_{metric}"] = b - a
            row[f"rel_diff_{metric}"] = (b - a) / a if pd.notna(a) and a != 0 else pd.NA

        rows.append(row)

    return pd.DataFrame(rows)


def best_and_worst_instances(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feasible = df[df["feasible"] == True].copy()

    if feasible.empty:
        sorted_df = df.sort_values(["objective", "total_routes"], ascending=[True, True])
        return sorted_df.head(n), sorted_df.tail(n)

    best = feasible.sort_values(["objective", "total_routes"], ascending=[True, True]).head(n)
    worst = feasible.sort_values(["objective", "total_routes"], ascending=[False, False]).head(n)

    return best, worst


def family_extremes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for family in FAMILY_ORDER:
        sub = df[df["family"] == family]
        if sub.empty:
            continue

        feasible = sub[sub["feasible"] == True]
        source = feasible if not feasible.empty else sub

        best = source.sort_values(["objective", "total_routes"], ascending=[True, True]).iloc[0]
        worst = source.sort_values(["objective", "total_routes"], ascending=[False, False]).iloc[0]

        rows.append(
            {
                "family": family,
                "best_instance": best["instance"],
                "best_objective": best["objective"],
                "best_routes": best["total_routes"],
                "worst_instance": worst["instance"],
                "worst_objective": worst["objective"],
                "worst_routes": worst["total_routes"],
            }
        )

    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    return df[cols].corr(numeric_only=True)


def build_insights(family_summary: pd.DataFrame, comparison: pd.DataFrame, overview: pd.DataFrame) -> List[str]:
    insights: List[str] = []

    if not family_summary.empty:
        best_family = family_summary.dropna(subset=["mean_objective"]).sort_values("mean_objective").iloc[0]
        worst_family = family_summary.dropna(subset=["mean_objective"]).sort_values("mean_objective", ascending=False).iloc[0]

        insights.append(f"Best average objective: {best_family['family']} with {best_family['mean_objective']:.2f}.")
        insights.append(f"Highest average objective: {worst_family['family']} with {worst_family['mean_objective']:.2f}.")

        best_feas = family_summary.sort_values("feasible_rate", ascending=False).iloc[0]
        insights.append(f"Highest feasibility rate: {best_feas['family']} at {best_feas['feasible_rate']:.2%}.")

    for _, row in comparison.iterrows():
        distribution = row["distribution"]
        diff_obj = row.get("diff_objective", pd.NA)
        diff_routes = row.get("diff_total_routes", pd.NA)
        diff_time = row.get("diff_total_time", pd.NA)

        if pd.notna(diff_obj):
            if diff_obj > 0:
                insights.append(
                    f"For {distribution}, tight instances are harder than large ones: objective increases by {diff_obj:.2f} on average."
                )
            else:
                insights.append(
                    f"For {distribution}, tight instances are not worse on objective on average: objective changes by {diff_obj:.2f}."
                )

        if pd.notna(diff_routes):
            insights.append(f"For {distribution}, route count changes by {diff_routes:.2f} from large to tight.")

        if pd.notna(diff_time):
            insights.append(f"For {distribution}, total time changes by {diff_time:.2f} from large to tight.")

    if not overview.empty:
        overall = overview.iloc[0]
        insights.append(
            f"Overall test feasible rate: {overall['feasible_rate']:.2%}, mean objective: {overall['mean_objective']:.2f}."
        )

    return insights


def write_markdown_report(
    output_path: Path,
    overview: pd.DataFrame,
    family_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    best: pd.DataFrame,
    worst: pd.DataFrame,
    extremes: pd.DataFrame,
    corr: pd.DataFrame,
    insights: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# VRPTW test CSV analysis")
    lines.append("")
    lines.append("## Key insights")
    for item in insights:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Overall test overview")
    lines.append(overview.to_markdown(index=False))
    lines.append("")
    lines.append("## Family summary")
    lines.append(family_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Large vs tight comparison inside each distribution")
    lines.append(comparison.to_markdown(index=False))
    lines.append("")
    lines.append("## Best instances")
    lines.append(best.to_markdown(index=False))
    lines.append("")
    lines.append("## Worst instances")
    lines.append(worst.to_markdown(index=False))
    lines.append("")
    lines.append("## Best and worst per family")
    lines.append(extremes.to_markdown(index=False))
    lines.append("")
    lines.append("## Correlation matrix")
    lines.append(corr.reset_index().rename(columns={"index": "metric"}).to_markdown(index=False))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple pandas analysis of the VRPTW test summary CSV.")
    parser.add_argument("--test_csv", required=True, help="Path to test_summary.csv")
    parser.add_argument("--output_dir", default="analysis_outputs/test_analysis", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = load_summary(args.test_csv)

    overview = split_overview(test_df)
    family_summary = aggregate_by_family(test_df)
    comparison = distribution_window_comparison(test_df)
    best, worst = best_and_worst_instances(test_df, n=5)
    extremes = family_extremes(test_df)
    corr = correlation_table(test_df)
    insights = build_insights(family_summary, comparison, overview)

    test_df.to_csv(out_dir / "test_combined.csv", index=False)
    overview.to_csv(out_dir / "test_overview.csv", index=False)
    family_summary.to_csv(out_dir / "test_family_summary.csv", index=False)
    comparison.to_csv(out_dir / "test_large_vs_tight_comparison.csv", index=False)
    best.to_csv(out_dir / "test_best_instances.csv", index=False)
    worst.to_csv(out_dir / "test_worst_instances.csv", index=False)
    extremes.to_csv(out_dir / "test_family_extremes.csv", index=False)
    corr.to_csv(out_dir / "test_correlation_matrix.csv")

    report_path = out_dir / "test_csv_analysis_report.md"
    write_markdown_report(
        output_path=report_path,
        overview=overview,
        family_summary=family_summary,
        comparison=comparison,
        best=best,
        worst=worst,
        extremes=extremes,
        corr=corr,
        insights=insights,
    )

    print(f"Saved report to: {report_path}")
    print(f"Saved tables to: {out_dir}")


if __name__ == "__main__":
    main()