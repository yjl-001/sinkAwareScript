import math
import statistics
from collections import defaultdict

from mvp.core.records import TriggerTracePointRecord


def summarize_trigger_sink_scores(points: list[TriggerTracePointRecord]) -> dict:
    """比较 Trigger 真正判定过的 inference candidates 的 first-key score。"""

    eligible = [
        point
        for point in points
        if point.point_type == "inference"
        and point.first_key_attention is not None
        and math.isfinite(point.first_key_attention)
    ]
    inserted = [point for point in eligible if point.actual_inserted]
    skipped = [point for point in eligible if not point.actual_inserted]
    inserted_scores = [float(point.first_key_attention) for point in inserted]
    skipped_scores = [float(point.first_key_attention) for point in skipped]

    paired_differences = _paired_sample_mean_differences(eligible)
    layer_windows = sorted(
        {
            point.sink_score_layer_window
            for point in eligible
            if point.sink_score_layer_window is not None
        }
    )
    return {
        "score_name": "first_key_attention",
        "layer_windows": layer_windows,
        "candidate_scope": "trigger-evaluated inference delimiter candidates before insertion",
        "eligible_candidate_count": len(eligible),
        "inserted": _group_summary(inserted),
        "not_inserted": _group_summary(skipped),
        "difference": {
            "pooled_mean_inserted_minus_not_inserted": _difference(
                _mean_or_none(inserted_scores), _mean_or_none(skipped_scores)
            ),
            "pooled_median_inserted_minus_not_inserted": _difference(
                _median_or_none(inserted_scores), _median_or_none(skipped_scores)
            ),
            "paired_sample_count": len(paired_differences),
            "paired_sample_mean_difference": _mean_or_none(paired_differences),
            "paired_sample_median_difference": _median_or_none(paired_differences),
        },
        "notes": [
            "Prompt candidates are excluded.",
            "Delimiters after the inference insertion budget is exhausted are not Trigger-evaluated candidates.",
            "Scores are measured before the candidate's latent insertion.",
            "Raw first-key attention can vary with context length; compare rel_pos statistics and the position plot.",
        ],
    }


def _group_summary(points: list[TriggerTracePointRecord]) -> dict:
    scores = [float(point.first_key_attention) for point in points]
    relative_positions = [float(point.rel_pos) for point in points]
    sample_count = len({point.sample_idx for point in points})
    return {
        "candidate_count": len(points),
        "sample_count": sample_count,
        "mean": _mean_or_none(scores),
        "median": _median_or_none(scores),
        "std": statistics.pstdev(scores) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "mean_rel_pos": _mean_or_none(relative_positions),
        "median_rel_pos": _median_or_none(relative_positions),
    }


def _paired_sample_mean_differences(points: list[TriggerTracePointRecord]) -> list[float]:
    grouped = defaultdict(lambda: {True: [], False: []})
    for point in points:
        grouped[point.sample_idx][point.actual_inserted].append(float(point.first_key_attention))
    return [
        statistics.fmean(groups[True]) - statistics.fmean(groups[False])
        for groups in grouped.values()
        if groups[True] and groups[False]
    ]


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None
