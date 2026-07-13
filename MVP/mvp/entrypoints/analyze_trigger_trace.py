#!/usr/bin/env python3
"""Regenerate Trigger sink-score statistics and figures from an existing trace."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

from mvp.core.records import TriggerTracePointRecord
from mvp.metrics.trigger_sink_comparison import summarize_trigger_sink_scores
from mvp.viz.trigger_probability_sink_viz import save_trigger_probability_sink_plot
from mvp.viz.trigger_sink_comparison_viz import save_trigger_sink_comparison
from mvp.viz.viz_io import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory produced by trigger_trace workflow.")
    parser.add_argument("--rows-path", default=None, help="Defaults to <output-dir>/trigger_trace_rows.jsonl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows_path = Path(args.rows_path) if args.rows_path else output_dir / "trigger_trace_rows.jsonl"
    points = load_points(rows_path)
    summary = summarize_trigger_sink_scores(points)

    summary_path = output_dir / "trigger_sink_score_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_trigger_sink_comparison(
        points,
        summary,
        output_dir / "trigger_sink_score_comparison.png",
    )
    save_trigger_probability_sink_plot(
        points,
        summary,
        output_dir / "trigger_probability_vs_sink_score.png",
    )
    print(f"Analyzed {summary['eligible_candidate_count']} inference candidates from {rows_path}")
    print(f"Wrote Trigger sink-score analysis to {output_dir}")


def load_points(path: Path) -> list[TriggerTracePointRecord]:
    field_names = {field.name for field in fields(TriggerTracePointRecord)}
    return [
        TriggerTracePointRecord(**{key: value for key, value in row.items() if key in field_names})
        for row in load_jsonl(path)
    ]


if __name__ == "__main__":
    main()
