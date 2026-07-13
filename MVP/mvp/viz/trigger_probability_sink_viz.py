import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mvp.core.records import TriggerTracePointRecord


def save_trigger_probability_sink_plot(points: list[TriggerTracePointRecord], summary: dict, path: Path) -> None:
    """展示 sink score 与 Trigger 概率、真实动作之间的关系。"""

    eligible = [
        point
        for point in points
        if point.point_type == "inference"
        and point.first_key_attention is not None
        and math.isfinite(point.first_key_attention)
        and math.isfinite(point.trigger_probability)
    ]
    inserted = [point for point in eligible if point.actual_inserted]
    skipped = [point for point in eligible if not point.actual_inserted]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for group, color, label in [
        (skipped, "#6b7280", "Not inserted"),
        (inserted, "#d1495b", "Inserted"),
    ]:
        axes[0].scatter(
            [float(point.first_key_attention) for point in group],
            [float(point.trigger_probability) for point in group],
            s=26,
            alpha=0.6,
            color=color,
            edgecolors="none",
            label=f"{label} (n={len(group)})",
        )
    axes[0].axhline(0.5, color="#2563eb", linewidth=1.2, linestyle="--", label="p=0.5")
    trend = _linear_trend(eligible)
    if trend is not None:
        x_min = min(float(point.first_key_attention) for point in eligible)
        x_max = max(float(point.first_key_attention) for point in eligible)
        slope, intercept = trend
        axes[0].plot(
            [x_min, x_max],
            [slope * x_min + intercept, slope * x_max + intercept],
            color="#111827",
            linewidth=1.5,
            label="linear trend",
        )
    axes[0].set_xlabel("first_key_attention")
    axes[0].set_ylabel("P(augment)")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Candidate-level Trigger probability")
    axes[0].grid(alpha=0.2)
    if eligible:
        axes[0].legend(fontsize=8)

    bins = _equal_count_bins(eligible, max_bins=4)
    if bins:
        xs = [statistics.fmean(float(point.first_key_attention) for point in group) for group in bins]
        mean_probabilities = [statistics.fmean(float(point.trigger_probability) for point in group) for group in bins]
        insertion_rates = [statistics.fmean(float(point.actual_inserted) for point in group) for group in bins]
        axes[1].plot(xs, mean_probabilities, marker="o", color="#2563eb", label="Mean P(augment)")
        axes[1].plot(xs, insertion_rates, marker="s", color="#d1495b", label="Actual insertion rate")
        for x_value, probability, group in zip(xs, mean_probabilities, bins):
            axes[1].annotate(f"n={len(group)}", (x_value, probability), xytext=(0, 7),
                             textcoords="offset points", ha="center", fontsize=8)
    axes[1].axhline(0.5, color="#6b7280", linewidth=1, linestyle="--")
    axes[1].set_xlabel("mean first_key_attention in equal-count bin")
    axes[1].set_ylabel("probability / insertion rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Sink-score quantile bins")
    axes[1].grid(alpha=0.2)
    if bins:
        axes[1].legend(fontsize=9)

    association = summary.get("trigger_probability_association", {})
    fig.suptitle(
        "Trigger probability vs pre-insertion first-key attention\n"
        f"Pearson={_format_metric(association.get('pearson'))}; "
        f"Spearman={_format_metric(association.get('spearman'))}; "
        f"within-sample Pearson={_format_metric(association.get('within_sample_centered_pearson'))}",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _equal_count_bins(points: list[TriggerTracePointRecord], max_bins: int) -> list[list[TriggerTracePointRecord]]:
    ordered = sorted(points, key=lambda point: float(point.first_key_attention))
    bin_count = min(max_bins, len(ordered))
    if bin_count == 0:
        return []
    return [
        ordered[index * len(ordered) // bin_count:(index + 1) * len(ordered) // bin_count]
        for index in range(bin_count)
    ]


def _linear_trend(points: list[TriggerTracePointRecord]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    xs = [float(point.first_key_attention) for point in points]
    ys = [float(point.trigger_probability) for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator < 1e-12:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    return slope, y_mean - slope * x_mean


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
