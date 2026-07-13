import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mvp.core.records import TriggerTracePointRecord


def save_trigger_sink_comparison(points: list[TriggerTracePointRecord], summary: dict, path: Path) -> None:
    """画 inserted/skip sink score 分布，并显式展示位置混杂。"""

    eligible = [
        point
        for point in points
        if point.point_type == "inference"
        and point.first_key_attention is not None
        and math.isfinite(point.first_key_attention)
    ]
    inserted = [point for point in eligible if point.actual_inserted]
    skipped = [point for point in eligible if not point.actual_inserted]
    groups = [skipped, inserted]
    colors = ["#6b7280", "#d1495b"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    score_groups = [[float(point.first_key_attention) for point in group] for group in groups]
    nonempty = [(index + 1, values) for index, values in enumerate(score_groups) if values]
    if nonempty:
        box = axes[0].boxplot(
            [values for _, values in nonempty],
            positions=[position for position, _ in nonempty],
            widths=0.45,
            patch_artist=True,
            showfliers=False,
        )
        for patch, (position, _) in zip(box["boxes"], nonempty):
            patch.set_facecolor(colors[position - 1])
            patch.set_alpha(0.25)
    for group_index, (group, color) in enumerate(zip(groups, colors), start=1):
        xs = [group_index + ((index % 11) - 5) * 0.014 for index in range(len(group))]
        ys = [float(point.first_key_attention) for point in group]
        axes[0].scatter(xs, ys, s=18, alpha=0.55, color=color, edgecolors="none")
    axes[0].set_xticks([1, 2], ["Not inserted", "Inserted"])
    axes[0].set_ylabel("first_key_attention")
    axes[0].set_title("Trigger candidate sink score")
    axes[0].grid(axis="y", alpha=0.2)

    for group, color, label in zip(groups, colors, ["Not inserted", "Inserted"]):
        axes[1].scatter(
            [point.rel_pos for point in group],
            [float(point.first_key_attention) for point in group],
            s=22,
            alpha=0.6,
            color=color,
            edgecolors="none",
            label=f"{label} (n={len(group)})",
        )
    axes[1].set_xlabel("relative generation position")
    axes[1].set_ylabel("first_key_attention")
    axes[1].set_title("Sink score vs candidate position")
    axes[1].grid(alpha=0.2)
    if eligible:
        axes[1].legend(fontsize=9)

    delta = summary["difference"]["pooled_mean_inserted_minus_not_inserted"]
    delta_text = "N/A" if delta is None else f"{delta:.6f}"
    layer_windows = {point.sink_score_layer_window for point in eligible}
    layer_text = str(next(iter(layer_windows))) if len(layer_windows) == 1 else "mixed"
    fig.suptitle(
        "Pre-insertion first-key attention at Trigger-evaluated delimiter candidates\n"
        f"mean(inserted) - mean(not inserted) = {delta_text}; layer_window={layer_text}",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
