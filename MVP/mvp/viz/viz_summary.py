from pathlib import Path

from mvp.viz.viz_io import family_means


def save_summary_report(summary: dict, fig_dir: Path, output_dir: Path) -> None:
    """写一个轻量 Markdown 报告，方便不打开图片时快速扫结果。"""

    reward = family_means(summary, "strategy_reward_mean")
    insertions = family_means(summary, "strategy_actual_insertions_mean")
    utility = flatten_candidate_metric(summary, "counterfactual_avg_utility_at_budget")
    utility_eff = flatten_candidate_metric(summary, "counterfactual_avg_utility_at_budget_effective")
    group_utility = flatten_group_metric(summary, "avg_utility")
    group_counts = flatten_group_metric(summary, "selected_count_mean")

    lines = [
        "# Sink-aware MVP Visualization Report",
        "",
        f"Source output: `{output_dir}`",
        "",
        "## Key Numbers",
        "",
        f"- Candidate rows: {summary.get('num_candidate_rows')}",
        f"- Sequence point rows: {summary.get('num_sequence_point_rows')}",
        f"- Selected point rows: {summary.get('num_selected_point_rows')}",
        f"- Strategy rows: {summary.get('num_strategy_rows')}",
        f"- References: {reference_keys(summary)}",
        "",
        "## Aggregates",
        "",
        f"- Reward means: `{reward}`",
        f"- Utility@B: `{utility}`",
        f"- Utility@B effective only: `{utility_eff}`",
        f"- Group avg utility: `{group_utility}`",
        f"- Group selected counts: `{group_counts}`",
        f"- Actual insertions: `{insertions}`",
    ]
    (fig_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def flatten_candidate_metric(summary: dict, metric: str) -> dict[str, float]:
    """读取旧 summary 中的 candidate selector 指标。"""

    values = {}
    for ref, payload in summary.get("candidate_by_reference", {}).items():
        for selector, value in payload.get(metric, {}).items():
            if value is not None:
                values[f"{ref}/{selector}"] = float(value)
    return values


def flatten_group_metric(summary: dict, metric: str) -> dict[str, float]:
    """读取新 summary 中的配置化实验组指标。"""

    values = {}
    for ref, payload in summary.get("counterfactual_groups_by_reference", {}).items():
        for group, metrics in payload.items():
            value = metrics.get(metric)
            if value is not None:
                values[f"{ref}/{group}"] = float(value)
    return values


def reference_keys(summary: dict) -> list[str]:
    """兼容旧/新 summary 的 reference key 展示。"""

    if "counterfactual_groups_by_reference" in summary:
        return list(summary.get("counterfactual_groups_by_reference", {}).keys())
    return list(summary.get("candidate_by_reference", {}).keys())
