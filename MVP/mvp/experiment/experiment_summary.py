def summarize_experiment(candidate_rows, sequence_rows, selected_rows, group_rows,
                         strategy_rows, experiment) -> dict:
    """汇总当前实验。

    主指标来自 group_rows：每个 sample/实验组先算一次 avg utility，再跨样本平均。
    因此第三组没有选中点的样本会以 avg_utility=0 参与统计。
    """

    summary = {
        "num_candidate_rows": len(candidate_rows),
        "num_sequence_point_rows": len(sequence_rows),
        "num_selected_point_rows": len(selected_rows),
        "num_group_rows": len(group_rows),
        "num_strategy_rows": len(strategy_rows),
        "experiment": {
            "reference_mode": experiment.reference_mode,
            "max_prompt_aug_num": experiment.max_prompt_aug_num,
            "first_key_layer_window": experiment.first_key_layer_window,
            "groups": [group.name for group in experiment.groups],
        },
        "counterfactual_groups_by_reference": {},
        "strategy_reward_mean": {},
        "strategy_actual_insertions_mean": {},
    }

    for ref in sorted({row.reference_mode for row in group_rows}):
        ref_rows = [row for row in group_rows if row.reference_mode == ref]
        ref_summary = {}
        for group in sorted({row.group for row in ref_rows}):
            rows = [row for row in ref_rows if row.group == group]
            ref_summary[group] = summarize_group(rows)
        summary["counterfactual_groups_by_reference"][ref] = ref_summary

    add_strategy_summary(summary, strategy_rows)
    return summary


def summarize_group(rows) -> dict:
    """汇总一个 reference mode 下一个实验组的 sample-level 结果。"""

    if not rows:
        return {}
    selected_counts = [row.selected_count for row in rows]
    inserted_counts = [row.inserted_count for row in rows]
    utilities = [row.avg_utility for row in rows]
    precisions = [row.positive_precision for row in rows]
    effective = [row.avg_utility for row in rows if row.avg_utility != 0.0]
    return {
        "num_samples": len(rows),
        "num_samples_with_selected_points": sum(count > 0 for count in selected_counts),
        "num_effective_samples": len(effective),
        "selected_count_mean": sum(selected_counts) / len(selected_counts),
        "inserted_count_mean": sum(inserted_counts) / len(inserted_counts),
        "avg_utility": sum(utilities) / len(utilities),
        "avg_utility_effective": sum(effective) / len(effective) if effective else None,
        "positive_precision": sum(precisions) / len(precisions),
    }


def add_strategy_summary(summary: dict, strategy_rows) -> None:
    """保留旧的策略级 rollout 汇总，后续线上策略实验可以继续接这里。"""

    for name in sorted({row.strategy for row in strategy_rows}):
        for ref in sorted({row.reference_mode for row in strategy_rows if row.strategy == name}):
            key = f"{ref}/{name}"
            rows = [row for row in strategy_rows if row.strategy == name and row.reference_mode == ref]
            rewards = [row.reward for row in rows]
            insertions = [len(row.inserted_steps) for row in rows]
            summary["strategy_reward_mean"][key] = sum(rewards) / len(rewards) if rewards else None
            summary["strategy_actual_insertions_mean"][key] = (
                sum(insertions) / len(insertions) if insertions else None
            )
