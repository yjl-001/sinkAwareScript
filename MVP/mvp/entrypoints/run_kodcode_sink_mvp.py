#!/usr/bin/env python3
"""Run the KodCode sink-aware latent-memory MVP."""

from pathlib import Path
import logging
import shutil

from omegaconf import OmegaConf

from mvp.config.cli import apply_overrides, parse_args, set_seed
from mvp.config.experiment_config import load_experiment_config
from mvp.core import repo_paths  # noqa: F401
from mvp.core.model_setup import build_dataset, encode_prompt, load_model, reward_completion
from mvp.core.records import TriggerTraceSampleRecord
from mvp.core.trigger_trace_generation import generate_with_trigger_trace
from mvp.experiment.candidate_experiment import run_reference, run_strategy_rollouts
from mvp.experiment.counterfactual_eval import evaluate_counterfactual_groups
from mvp.experiment.experiment_summary import summarize_experiment
from mvp.io.outputs import append_strategy_csv, write_dataclass_jsonl, write_summary
from mvp.metrics.trigger_sink_comparison import summarize_trigger_sink_scores
from mvp.viz.trigger_sink_comparison_viz import save_trigger_sink_comparison
from mvp.viz.trigger_trace_viz import save_trigger_trace_visuals


LOGGER = logging.getLogger("sink-aware-mvp")


def prepare_output_paths(output_dir: Path) -> dict[str, Path]:
    """集中管理输出路径，避免主流程里散落字符串。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "candidate": output_dir / "candidate_rows.jsonl",
        "sequence": output_dir / "sequence_points.jsonl",
        "selected": output_dir / "selected_point_rows.jsonl",
        "group": output_dir / "group_rows.jsonl",
        "strategy": output_dir / "strategy_rows.jsonl",
        "summary": output_dir / "summary.json",
        "strategy_csv": output_dir / "strategy_rows.csv",
        "sink_events": output_dir / "sink_events.jsonl",
        "trigger_trace": output_dir / "trigger_trace_rows.jsonl",
        "trigger_samples": output_dir / "trigger_trace_samples.jsonl",
        "trigger_sink_summary": output_dir / "trigger_sink_score_summary.json",
        "trigger_sink_figure": output_dir / "trigger_sink_score_comparison.png",
    }


def maybe_clear_outputs(paths: dict[str, Path], output_dir: Path, overwrite: bool) -> None:
    """避免复用 output-dir 时 JSONL/CSV 追加污染新实验。"""

    if not overwrite:
        return
    for path in paths.values():
        if path.exists():
            path.unlink()
    for dir_name in ["figures", "attention_heatmaps", "trigger_trace_heatmaps"]:
        path = output_dir / dir_name
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    paths = prepare_output_paths(output_dir)
    maybe_clear_outputs(paths, output_dir, args.overwrite)
    config = apply_overrides(OmegaConf.load(args.cfg_path), args.options)
    workflow = getattr(args, "workflow", "candidate")
    validate_workflow_args(workflow, args)
    experiment = load_experiment_config(args.experiment_config) if workflow == "candidate" else None
    if experiment is not None:
        args.first_key_layer_window = experiment.first_key_layer_window
    config.dataset.mode = "sft"
    if experiment is not None:
        config.model.max_prompt_aug_num = experiment.max_prompt_aug_num
    config.model.max_inference_aug_num = args.budget

    LOGGER.info("Loading KodCode dataset...")
    dataset = build_dataset(config, args.split)
    end = min(args.start_index + args.limit, len(dataset))
    sample_indices = list(range(args.start_index, end))

    LOGGER.info("Loading MemGen model...")
    model = load_model(config, args)
    if workflow == "trigger_trace":
        run_trigger_trace(model, dataset, sample_indices, paths, args)
        LOGGER.info("Done. Trigger trace outputs written to %s", args.output_dir)
        return
    if workflow != "candidate":
        raise ValueError(f"Unsupported MVP workflow: {workflow}")

    all_candidate_rows = []
    all_sequence_rows = []
    all_selected_rows = []
    all_group_rows = []
    all_strategy_rows = []

    for local_idx, sample_idx in enumerate(sample_indices, start=1):
        sample = dataset[sample_idx]
        LOGGER.info("Sample %s/%s: dataset index %s", local_idx, len(sample_indices), sample_idx)
        prompt_ids, prompt_mask = encode_prompt(model, sample, args.device)

        reference_mode = experiment.reference_mode
        baseline, reference_reward, candidates = run_reference(
            model, prompt_ids, prompt_mask, sample, sample_idx, experiment, args
        )
        if not candidates:
            LOGGER.info("No delimiter candidates found for sample=%s ref=%s", sample_idx, reference_mode)

        selected_rows, group_rows = evaluate_counterfactual_groups(
            model, prompt_ids, prompt_mask, sample, sample_idx, reference_mode,
            reference_reward, candidates, baseline.sequence_points, experiment, args
        )
        strategy_rows = []
        if args.run_strategy_rollouts:
            strategy_rows = run_strategy_rollouts(
                model, prompt_ids, prompt_mask, sample, sample_idx,
                reference_mode, baseline, reference_reward, candidates, experiment, args
            )

        all_candidate_rows.extend(candidates)
        all_sequence_rows.extend(baseline.sequence_points)
        all_selected_rows.extend(selected_rows)
        all_group_rows.extend(group_rows)
        all_strategy_rows.extend(strategy_rows)
        write_dataclass_jsonl(paths["candidate"], candidates)
        write_dataclass_jsonl(paths["sequence"], baseline.sequence_points)
        write_dataclass_jsonl(paths["selected"], selected_rows)
        write_dataclass_jsonl(paths["group"], group_rows)
        if strategy_rows:
            write_dataclass_jsonl(paths["strategy"], strategy_rows)
            append_strategy_csv(paths["strategy_csv"], strategy_rows)

        summary = summarize_experiment(
            all_candidate_rows, all_sequence_rows, all_selected_rows,
            all_group_rows, all_strategy_rows, experiment
        )
        write_summary(paths["summary"], summary)
        LOGGER.info("Running summary: %s", summary)

    LOGGER.info("Done. Outputs written to %s", args.output_dir)


def validate_workflow_args(workflow: str, args) -> None:
    if workflow != "trigger_trace":
        return
    if not args.load_model_path:
        raise ValueError("trigger_trace requires --load-model-path pointing to a trained Trigger checkpoint")
    if args.attn_implementation != "eager":
        raise ValueError("trigger_trace requires attn_implementation=eager to obtain attention heatmaps")
    trace_config = getattr(args, "trigger_trace", {}) or {}
    if float(trace_config.get("trigger_temperature", 1.0)) < 0:
        raise ValueError("trigger_trace.trigger_temperature must be non-negative")
    if int(trace_config.get("heatmap_layer_window", 4)) < 0:
        raise ValueError("trigger_trace.heatmap_layer_window must be >= 0")
    if int(trace_config.get("sink_score_layer_window", 4)) < 0:
        raise ValueError("trigger_trace.sink_score_layer_window must be >= 0")


def run_trigger_trace(model, dataset, sample_indices: list[int], paths: dict[str, Path], args) -> None:
    """运行训练后 Trigger 的在线轨迹，并保存插入前 attention heatmap。"""

    trace_config = getattr(args, "trigger_trace", {}) or {}
    all_points = []
    for local_idx, sample_idx in enumerate(sample_indices, start=1):
        sample = dataset[sample_idx]
        LOGGER.info("Trigger trace %s/%s: dataset index %s", local_idx, len(sample_indices), sample_idx)
        prompt_ids, prompt_mask = encode_prompt(model, sample, args.device)
        trace = generate_with_trigger_trace(
            model,
            prompt_ids,
            prompt_mask,
            sample_idx=sample_idx,
            args=args,
            trace_config=trace_config,
        )
        reward = reward_completion(trace.completion, sample)
        for point in trace.points:
            point.reward = reward

        sample_record = TriggerTraceSampleRecord(
            sample_idx=sample_idx,
            reward=reward,
            generated_len=len(trace.completion_ids),
            prompt_inserted=trace.prompt_inserted,
            inference_inserted_count=trace.inference_inserted_count,
            completion=trace.completion,
        )
        sample_record.contact_sheet_path = save_trigger_trace_visuals(
            trace.snapshots,
            sample_record,
            args,
            trace_config,
        )
        write_dataclass_jsonl(paths["trigger_trace"], trace.points)
        write_dataclass_jsonl(paths["trigger_samples"], [sample_record])
        all_points.extend(trace.points)
        if bool(trace_config.get("collect_candidate_sink_scores", True)):
            sink_summary = summarize_trigger_sink_scores(all_points)
            write_summary(paths["trigger_sink_summary"], sink_summary)
        LOGGER.info(
            "Trigger trace sample=%s reward=%.4f prompt_inserted=%s inference_insertions=%s heatmaps=%s",
            sample_idx,
            reward,
            trace.prompt_inserted,
            trace.inference_inserted_count,
            len(trace.snapshots),
        )

    if bool(trace_config.get("collect_candidate_sink_scores", True)):
        sink_summary = summarize_trigger_sink_scores(all_points)
        if bool(trace_config.get("save_sink_score_comparison_plot", True)):
            save_trigger_sink_comparison(
                all_points,
                sink_summary,
                paths["trigger_sink_figure"],
            )
        LOGGER.info("Trigger sink score comparison: %s", sink_summary["difference"])
if __name__ == "__main__":
    main()
