from pathlib import Path

import matplotlib.pyplot as plt

from mvp.viz.plot_utils import save_bar
from mvp.viz.selected_point_stats import group_rows


def save_selected_point_plots(rows: list[dict], summary: dict[str, dict], out_dir: Path) -> None:
    """保存 selected point 审计图。"""

    save_metric_bars(summary, out_dir)
    grouped = group_rows(rows)
    save_group_histograms(grouped, "utility", "Selected Point Utility", out_dir / "utility_by_group.png")
    save_group_histograms(grouped, "score", "Selected Point Score", out_dir / "score_by_group.png")
    save_group_histograms(grouped, "step", "Selected Point Step", out_dir / "step_by_group.png")
    save_score_utility_scatter(grouped, out_dir / "score_vs_utility_by_group.png")
    save_step_utility_scatter(grouped, out_dir / "step_vs_utility_by_group.png")


def save_metric_bars(summary: dict[str, dict], out_dir: Path) -> None:
    """保存组级统计条形图。"""

    for metric, title, ylabel, filename in [
        ("avg_utility_point_level", "Point-level Avg Utility", "utility", "point_avg_utility.png"),
        ("positive_precision_point_level", "Point-level Positive Precision", "precision", "point_positive_precision.png"),
        ("delimiter_fraction", "Selected Points After Delimiter", "fraction", "delimiter_fraction.png"),
        ("inserted_fraction", "Inserted Fraction", "fraction", "inserted_fraction.png"),
        ("step_mean", "Mean Selected Step", "step", "mean_step.png"),
        ("source_rank_mean", "Mean Source Rank", "rank", "mean_source_rank.png"),
    ]:
        values = {group: metrics.get(metric, 0.0) for group, metrics in summary.items()}
        save_bar(values, title, ylabel, out_dir / filename)


def save_group_histograms(grouped: dict[str, list[dict]], field: str, title: str, path: Path) -> None:
    """把每个 group 的字段分布画在同一张直方图里。"""

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for group, rows in grouped.items():
        values = [float(row.get(field, 0.0) or 0.0) for row in rows]
        if values:
            ax.hist(values, bins=20, alpha=0.45, label=group)
    ax.axvline(0, color="#444444", linewidth=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(field)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_score_utility_scatter(grouped: dict[str, list[dict]], path: Path) -> None:
    """画 score 与 utility 的关系，观察 sink score 是否真的对应收益。"""

    save_two_field_scatter(grouped, "score", "utility", "Score vs Utility", "score", "utility", path)


def save_step_utility_scatter(grouped: dict[str, list[dict]], path: Path) -> None:
    """画 step 与 utility 的关系，观察是否主要是位置效应。"""

    save_two_field_scatter(grouped, "step", "utility", "Step vs Utility", "step", "utility", path)


def save_two_field_scatter(grouped: dict[str, list[dict]], x_field: str, y_field: str,
                           title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, rows in grouped.items():
        xs = [float(row.get(x_field, 0.0) or 0.0) for row in rows]
        ys = [float(row.get(y_field, 0.0) or 0.0) for row in rows]
        ax.scatter(xs, ys, s=24, alpha=0.65, label=group, edgecolors="none")
    ax.axhline(0, color="#444444", linewidth=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
