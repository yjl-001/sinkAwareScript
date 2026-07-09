from mvp.config.experiment_config import GroupConfig


def select_points_for_group(group: GroupConfig, candidates, sequence_points):
    """根据 group 配置从 candidate 或 sequence pool 中选点。

    返回的是原始 record 对象列表。后续反事实评估会再把它们包装成
    SelectedPointRecord，这样 selector 本身不关心 reward/utility。
    """

    pool = candidates if group.source == "candidate" else sequence_points
    if group.selector == "first_k":
        return pool[: int(group.budget or len(pool))]
    if group.selector == "top_k":
        return sorted(pool, key=lambda row: score_of(row, group.score), reverse=True)[: int(group.budget or len(pool))]
    if group.selector == "threshold":
        threshold = float(group.threshold if group.threshold is not None else 0.0)
        return [row for row in pool if score_of(row, group.score) > threshold]
    raise ValueError(f"Unsupported selector: {group.selector}")


def score_of(row, score_name: str) -> float:
    """从 record 上读取分数，缺失或非数值时给极小值。"""

    value = getattr(row, score_name, float("-inf"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")
