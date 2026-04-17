from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go

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


def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_routes(payload: Dict[str, Any]) -> List[List[int]]:
    routes: List[List[int]] = []
    constraints = payload.get("constraints", {})
    route_details = constraints.get("route_details", [])

    for item in route_details:
        if isinstance(item, list):
            routes.append([int(x) for x in item])
        elif isinstance(item, dict) and "path" in item:
            routes.append([int(x) for x in item["path"]])

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
    xs = [instance.all_nodes[n].x for n in route]
    ys = [instance.all_nodes[n].y for n in route]
    if not xs:
        return 0.0, 0.0
    mid = len(xs) // 2
    return xs[mid], ys[mid]


def build_summary_table(payload: Dict[str, Any]) -> str:
    constraints = payload.get("constraints", {})
    evaluation = payload.get("evaluation", {})
    errors = constraints.get("errors", [])
    missing = constraints.get("missing_customers", [])
    duplicate = constraints.get("duplicate_customers", [])

    rows = [
        ("Instance", payload.get("instance", "Unknown")),
        ("Feasible", str(payload.get("feasible", False))),
        ("Routes used", str(constraints.get("vehicles_used", 0))),
        ("Total routes", str(constraints.get("total_routes", 0))),
        ("Total distance", f"{evaluation.get('total_distance', 0.0):.3f}"),
        ("Total time", f"{evaluation.get('total_time', 0.0):.3f}"),
        ("Load variance", f"{evaluation.get('load_variance', 0.0):.3f}"),
        ("Spatial variance", f"{evaluation.get('spatial_variance', 0.0):.3f}"),
        ("Objective", f"{evaluation.get('objective', 0.0):.3f}"),
        ("Errors", "; ".join(errors) if errors else "None"),
        ("Missing customers", ", ".join(map(str, missing)) if missing else "None"),
        ("Duplicate customers", ", ".join(map(str, duplicate)) if duplicate else "None"),
    ]

    html = [
        "<table style='width:100%; border-collapse:collapse; font-family:Arial,sans-serif; font-size:14px;'>",
        "<tbody>",
    ]
    for k, v in rows:
        html.append(
            f"<tr>"
            f"<th style='text-align:left; padding:8px; border-bottom:1px solid #ddd; width:220px; background:#fafafa;'>{k}</th>"
            f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{v}</td>"
            f"</tr>"
        )
    html.append("</tbody></table>")
    return "\n".join(html)


def build_route_figure(instance_txt: str, routes: List[List[int]], payload: Dict[str, Any]) -> go.Figure:
    instance = parse_instance(instance_txt)
    nodes = instance.all_nodes

    fig = go.Figure()

    customer_x = [n.x for n in nodes[1:]]
    customer_y = [n.y for n in nodes[1:]]

    fig.add_trace(
        go.Scatter(
            x=customer_x,
            y=customer_y,
            mode="markers",
            marker=dict(size=6, color="rgba(80,80,80,0.18)"),
            hoverinfo="skip",
            name="Customers",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[nodes[0].x],
            y=[nodes[0].y],
            mode="markers+text",
            marker=dict(size=18, symbol="star", color="#f58518", line=dict(color="black", width=1)),
            text=["Depot"],
            textposition="top center",
            hovertemplate="Depot<extra></extra>",
            name="Depot",
        )
    )

    for idx, route in enumerate(routes):
        if len(route) < 2:
            continue

        color = PALETTE[idx % len(PALETTE)]
        x = [nodes[n].x for n in route]
        y = [nodes[n].y for n in route]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(color=color, width=5),
                marker=dict(size=8, color=color),
                text=[str(n) for n in route],
                hovertemplate=(
                    f"<b>Vehicle {idx + 1}</b>"
                    "<br>node=%{text}"
                    f"<br>distance={route_distance(route, instance):.2f}"
                    f"<br>load={route_load(route, instance):.1f}"
                    "<extra></extra>"
                ),
                name=f"Vehicle {idx + 1}",
            )
        )

        mid_x, mid_y = route_midpoint(route, instance)
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            text=f"V{idx + 1}",
            showarrow=False,
            font=dict(size=12, color=color),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color,
            borderwidth=1,
            borderpad=2,
        )

    fig.update_layout(
        title=f"ALNS route solution — {payload.get('instance', instance.name)}",
        template="plotly_white",
        width=1500,
        height=1000,
        margin=dict(l=30, r=30, t=80, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        font=dict(size=14),
        title_x=0.5,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)

    return fig


def build_html_report(payload: Dict[str, Any], fig: go.Figure) -> str:
    summary_html = build_summary_table(payload)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>VRPTW Clustered Tight Showcase</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 24px;
            background: #ffffff;
            color: #111;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .card {{
            border: 1px solid #e6e6e6;
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 22px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 30px;
        }}
        h2 {{
            margin-top: 0;
            font-size: 20px;
        }}
        .muted {{
            color: #666;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>VRPTW solution showcase</h1>
        <div class="muted">Single clustered-tight test instance with a clean ALNS route visualization.</div>
    </div>

    <div class="card">
        <h2>Solution details</h2>
        {summary_html}
    </div>

    <div class="card">
        <h2>Route visualization</h2>
        {fig_html}
    </div>
</div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean route showcase for one clustered-tight VRPTW instance.")
    parser.add_argument("--instance_txt", required=True, help="Path to the instance .TXT file")
    parser.add_argument("--instance_json", required=True, help="Path to the matching solution JSON file")
    parser.add_argument("--output_html", default="analysis_outputs/clustered_tight_showcase.html", help="Output HTML path")
    args = parser.parse_args()

    payload = load_json(args.instance_json)
    routes = extract_routes(payload)

    if not routes:
        raise ValueError("No routes found in JSON file.")

    fig = build_route_figure(args.instance_txt, routes, payload)
    html = build_html_report(payload, fig)

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"Saved showcase to: {output_path}")


if __name__ == "__main__":
    main()