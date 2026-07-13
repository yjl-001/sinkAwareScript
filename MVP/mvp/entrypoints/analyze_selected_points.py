#!/usr/bin/env python3
"""Analyze selected_point_rows.jsonl from a sink-aware MVP run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from mvp.viz.selected_point_plots import save_selected_point_plots
from mvp.viz.selected_point_stats import sample_group_matrix, summarize_selected_points
from mvp.viz.viz_io import ensure_dir, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory produced by run_kodcode_sink_mvp.py.")
    parser.add_argument("--fig-dir", default=None, help="Defaults to <output-dir>/selected_point_figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else output_dir / "selected_point_figures"
    ensure_dir(fig_dir)

    rows = load_jsonl(output_dir / "selected_point_rows.jsonl")
    rows = [row for row in rows if row.get("utility") is not None]
    summary = summarize_selected_points(rows)
    save_selected_point_plots(rows, summary, fig_dir)
    save_report(summary, rows, fig_dir, output_dir)
    (fig_dir / "selected_point_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote selected point analysis to {fig_dir}")


def save_report(summary: dict[str, dict], rows: list[dict], fig_dir: Path, output_dir: Path) -> None:
    """写 Markdown 报告，列出关键指标和图像清单。"""

    lines = [
        "# Selected Point Analysis",
        "",
        f"Source output: `{output_dir}`",
        f"Selected rows: `{len(rows)}`",
        "",
        "## Group Metrics",
        "",
        "| group | points | avg U | pos precision | delimiter frac | inserted frac | mean step | mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, metrics in summary.items():
        lines.append(format_metric_row(group, metrics))

    lines.extend([
        "",
        "## Figures",
        "",
        "- `point_avg_utility.png`: point-level average utility by group.",
        "- `point_positive_precision.png`: fraction of selected points with positive utility.",
        "- `delimiter_fraction.png`: fraction selected from delimiter-after positions.",
        "- `inserted_fraction.png`: whether selected points actually inserted.",
        "- `mean_step.png`: average selected generation step.",
        "- `mean_source_rank.png`: average rank inside the source pool.",
        "- `utility_by_group.png`: utility distribution by group.",
        "- `score_by_group.png`: selector score distribution by group.",
        "- `step_by_group.png`: selected step distribution by group.",
        "- `score_vs_utility_by_group.png`: selector score vs utility.",
        "- `step_vs_utility_by_group.png`: position vs utility.",
        "",
        "## Per-sample Avg Utility",
        "",
    ])
    lines.extend(format_sample_matrix(sample_group_matrix(rows, "utility")))
    (fig_dir / "selected_point_report.md").write_text("\n".join(lines), encoding="utf-8")


def format_metric_row(group: str, metrics: dict) -> str:
    return (
        f"| {group} | {metrics.get('num_points', 0)} | "
        f"{metrics.get('avg_utility_point_level', 0.0):.4f} | "
        f"{metrics.get('positive_precision_point_level', 0.0):.4f} | "
        f"{metrics.get('delimiter_fraction', 0.0):.4f} | "
        f"{metrics.get('inserted_fraction', 0.0):.4f} | "
        f"{metrics.get('step_mean', 0.0):.2f} | "
        f"{metrics.get('source_rank_mean', 0.0):.2f} |"
    )


def format_sample_matrix(matrix: dict[str, dict[int, float]]) -> list[str]:
    """把每个 sample 的组内平均 utility 写成表，方便定位异常样本。"""

    sample_ids = sorted({idx for group_values in matrix.values() for idx in group_values})
    groups = sorted(matrix)
    lines = ["| sample_idx | " + " | ".join(groups) + " |"]
    lines.append("|---:|" + "|".join(["---:"] * len(groups)) + "|")
    for sample_idx in sample_ids:
        values = [matrix[group].get(sample_idx, 0.0) for group in groups]
        lines.append(f"| {sample_idx} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    return lines


if __name__ == "__main__":
    main() 
