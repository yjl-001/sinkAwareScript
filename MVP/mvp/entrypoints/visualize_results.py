#!/usr/bin/env python3
"""Create plots for a sink-aware MVP output directory."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from mvp.viz.plot_utils import save_bar, save_hist, save_scatter
from mvp.viz.viz_io import ensure_dir, family_means, load_json, load_jsonl
from mvp.viz.viz_summary import flatten_candidate_metric, flatten_group_metric, save_summary_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory produced by run_kodcode_sink_mvp.py.")
    parser.add_argument("--fig-dir", default=None, help="Defaults to <output-dir>/figures.")
    return parser.parse_args()


def finite_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if row.get("utility") is not None
        and row.get("sink_mass_z") is not None
        and row.get("entropy_z") is not None
    ]


def finite_selected_rows(rows: list[dict]) -> list[dict]:
    """筛出已经完成单点反事实评估的 selected point。"""

    return [
        row for row in rows
        if row.get("utility") is not None and row.get("score") is not None
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else output_dir / "figures"
    ensure_dir(fig_dir)

    summary = load_json(output_dir / "summary.json")
    candidates = finite_rows(load_jsonl(output_dir / "candidate_rows.jsonl"))
    selected = finite_selected_rows(load_jsonl(output_dir / "selected_point_rows.jsonl"))

    save_bar(
        family_means(summary, "strategy_reward_mean"),
        "Strategy Reward Mean",
        "reward",
        fig_dir / "strategy_reward_mean.png",
    )
    save_bar(
        flatten_candidate_metric(summary, "counterfactual_avg_utility_at_budget"),
        "Counterfactual Avg Utility@Budget",
        "Avg U@B",
        fig_dir / "counterfactual_utility_at_budget.png",
    )
    save_bar(
        flatten_candidate_metric(summary, "counterfactual_avg_utility_at_budget_effective"),
        "Counterfactual Avg Utility@Budget (Effective Samples)",
        "Avg U@B",
        fig_dir / "counterfactual_utility_at_budget_effective.png",
    )
    save_bar(
        flatten_group_metric(summary, "avg_utility"),
        "Configured Groups Avg Utility",
        "avg utility",
        fig_dir / "counterfactual_group_avg_utility.png",
    )
    save_bar(
        flatten_group_metric(summary, "selected_count_mean"),
        "Configured Groups Selected Count Mean",
        "selected points",
        fig_dir / "counterfactual_group_selected_count.png",
    )
    save_bar(
        family_means(summary, "strategy_actual_insertions_mean"),
        "Actual Insertions Mean",
        "actual insertions",
        fig_dir / "actual_insertions_mean.png",
    )

    if candidates:
        utilities = [float(row["utility"]) for row in candidates]
        pos_colors = [float(row.get("rel_pos", 0.0)) for row in candidates]
        save_scatter(
            [float(row["sink_mass_z"]) for row in candidates],
            utilities,
            pos_colors,
            "SinkMassZ vs Candidate Utility",
            "SinkMassZ",
            "U(j)",
            fig_dir / "sink_z_vs_utility.png",
        )
        save_scatter(
            [float(row["entropy_z"]) for row in candidates],
            utilities,
            pos_colors,
            "EntropyZ vs Candidate Utility",
            "EntropyZ",
            "U(j)",
            fig_dir / "entropy_z_vs_utility.png",
        )
        save_hist(utilities, "Candidate Utility Distribution", "U(j)", fig_dir / "utility_hist.png")

    if selected:
        save_scatter(
            [float(row["score"]) for row in selected],
            [float(row["utility"]) for row in selected],
            [float(row.get("rank_in_group", 0.0)) for row in selected],
            "Selected Point Score vs Utility",
            "selector score",
            "U(j)",
            fig_dir / "selected_score_vs_utility.png",
        )
        save_hist(
            [float(row["utility"]) for row in selected],
            "Selected Point Utility Distribution",
            "U(j)",
            fig_dir / "selected_utility_hist.png",
        )

    save_summary_report(summary, fig_dir, output_dir)
    print(f"Wrote figures and report to {fig_dir}")


if __name__ == "__main__":
    main()
