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
    sink_scores = [float(point.first_key_attention) for point in eligible]
    trigger_probabilities = [float(point.trigger_probability) for point in eligible]

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
        "trigger_probability_association": {
            "pearson": _pearson(sink_scores, trigger_probabilities),
            "spearman": _spearman(sink_scores, trigger_probabilities),
            "within_sample_centered_pearson": _within_sample_centered_pearson(eligible),
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


def _within_sample_centered_pearson(points: list[TriggerTracePointRecord]) -> float | None:
    grouped = defaultdict(list)
    for point in points:
        grouped[point.sample_idx].append(point)

    centered_scores = []
    centered_probabilities = []
    for sample_points in grouped.values():
        if len(sample_points) < 2:
            continue
        scores = [float(point.first_key_attention) for point in sample_points]
        probabilities = [float(point.trigger_probability) for point in sample_points]
        score_mean = statistics.fmean(scores)
        probability_mean = statistics.fmean(probabilities)
        centered_scores.extend(score - score_mean for score in scores)
        centered_probabilities.extend(probability - probability_mean for probability in probabilities)
    return _pearson(centered_scores, centered_probabilities)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in range(start, end):
            ranks[ordered[index][0]] = average_rank
        start = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator < 1e-12:
        return None
    numerator = sum(x_value * y_value for x_value, y_value in zip(x_centered, y_centered))
    return numerator / denominator
