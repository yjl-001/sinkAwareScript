from collections import defaultdict


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """按实验组聚合 selected_point_rows。"""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group", "unknown"))].append(row)
    return dict(grouped)


def summarize_selected_points(rows: list[dict]) -> dict[str, dict]:
    """计算 selected_point_rows 的组级统计。

    这里是 point-level 统计，和 summary.json 中 sample-level avg utility 不同。
    它回答的是：这个组实际选出来的所有点整体长什么样。
    """

    summary = {}
    for group, items in group_rows(rows).items():
        utilities = [as_float(row.get("utility")) for row in items]
        scores = [as_float(row.get("score")) for row in items]
        steps = [as_float(row.get("step")) for row in items]
        source_ranks = [as_float(row.get("source_rank")) for row in items]
        summary[group] = {
            "num_points": len(items),
            "avg_utility_point_level": mean(utilities),
            "positive_precision_point_level": mean([value > 0.0 for value in utilities]),
            "zero_utility_fraction": mean([value == 0.0 for value in utilities]),
            "inserted_fraction": mean([bool(row.get("branch_inserted")) for row in items]),
            "delimiter_fraction": mean([bool(row.get("source_prefix_ends_with_delimiter")) for row in items]),
            "score_mean": mean(scores),
            "score_max": max(scores) if scores else 0.0,
            "step_mean": mean(steps),
            "source_rank_mean": mean(source_ranks),
        }
    return summary


def sample_group_matrix(rows: list[dict], value_key: str) -> dict[str, dict[int, float]]:
    """按 group/sample 聚合一个字段，方便观察每个样本内的平均情况。"""

    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        group = str(row.get("group", "unknown"))
        sample_idx = int(row.get("sample_idx", -1))
        buckets[(group, sample_idx)].append(as_float(row.get(value_key)))

    matrix: dict[str, dict[int, float]] = defaultdict(dict)
    for (group, sample_idx), values in buckets.items():
        matrix[group][sample_idx] = mean(values)
    return dict(matrix)


def as_float(value) -> float:
    """把 JSON 中的数字安全转成 float；缺失时按 0 处理。"""

    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values) -> float:
    """普通均值；空列表返回 0，方便绘图。"""

    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else 0.0
