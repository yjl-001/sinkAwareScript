# Sink-aware MVP for KodCode

This directory contains a narrow validation script for the question:

```text
Does first-token sink attention identify useful latent-memory insertion points,
compared with the first-K delimiter baseline?
```

The script is intentionally standalone. It does not modify the MemGen training or
generation code paths.

## Module Layout

The root Python files are compatibility wrappers. Implementation lives under
`mvp/`:

- `mvp/config/`: CLI shim plus YAML-backed run/experiment configuration.
- `mvp/core/`: model loading, controlled generation, shared records.
- `mvp/metrics/`: attention and uncertainty metrics.
- `mvp/experiment/`: point selection, counterfactual evaluation, summaries.
- `mvp/io/`: JSONL/CSV/summary writers.
- `mvp/viz/`: attention heatmaps and result plotting helpers.
- `mvp/entrypoints/`: packaged entrypoints called by the root wrappers.

Config files live under `configs/`:

- `run_kodcode_default.yaml`: main run/model/generation/evaluation settings.
- `run_kodcode_trigger_trace.yaml`: trained-Trigger online insertion heatmap run.
- `first_key_sink_three_groups.yaml`: prompt augmentation and group selectors.
- `viz_default.yaml`: heatmap and sink-event visualization settings.
- `trigger_trace_default.yaml`: Trigger decision and trace-visualization settings.

## What It Runs

For each KodCode sample, the script:

1. Loads a MemGen model and a Weaver checkpoint.
2. Reads the experiment config to decide the reference mode:
   - `max_prompt_aug_num: 1`: insert prompt latent before decoding.
   - `max_prompt_aug_num: 0`: no prompt latent.
3. Runs one reference generation and records:
   - delimiter candidates;
   - every baseline step's first-key sink score.
4. Selects points according to the configured groups.
5. Runs single-insertion counterfactual branches for selected points:

```text
U(j) = reward(force one inference latent at step j) - reward(reference)
```

The default config compares:

```text
candidate_first_k
candidate_first_key_sink_top5
sequence_first_key_sink_threshold_0_5
```

The third group may insert at non-delimiter positions. If it selects no point
for a sample, that sample contributes `avg_utility=0` for the group.

6. Optionally runs exploratory budget-matched strategy rollouts with
   `--run-strategy-rollouts`:

```text
first_k
random
same_bucket_random
sink_top_b
sink_entropy_top_b
```

## Example

```bash
conda activate memgen
python sinkAwareScript/MVP/run_kodcode_sink_mvp.py \
  --run-config sinkAwareScript/MVP/configs/run_kodcode_default.yaml \
  --load-model-path MemGen/Qwen2.5-1.5B-Instruct/kodcode/weaver-sft/pn=1_pl=4_in=5_il=4/model \
  --output-dir output/sink_aware_mvp/kodcode_debug \
  --experiment-config sinkAwareScript/MVP/configs/first_key_sink_three_groups.yaml \
  --limit 20 \
  --overwrite
```

Use a small `--limit` first. The counterfactual branches are expensive because
each selected point can trigger an additional generation.

## Trained Trigger Insertion Heatmaps

The `trigger_trace` workflow audits where a trained Trigger actually inserts
memory during an online rollout. Its primary artifacts remain pre-insertion
attention heatmaps. It does not compute SinkMass, SinkMassZ, thresholds, or
rankings; the optional candidate comparison uses only `first_key_attention`.

Run it with the final Trigger checkpoint:

```bash
conda activate memgen
python sinkAwareScript/MVP/run_kodcode_sink_mvp.py \
  --run-config sinkAwareScript/MVP/configs/run_kodcode_trigger_trace.yaml \
  --load-model-path /path/to/trained-trigger/model \
  --output-dir output/sink_aware_mvp/kodcode_trigger_trace_debug \
  --limit 20 \
  --overwrite
```

The run config selects `workflow: trigger_trace`; behavior and image limits are
kept in `configs/trigger_trace_default.yaml`. The default policy audit uses:

```yaml
trigger_trace:
  trigger_active: true
  trigger_do_sample: false
  trigger_temperature: 1.0
  collect_candidate_sink_scores: true
  sink_score_layer_window: 4
  save_sink_score_comparison_plot: true
  save_prompt_heatmap: false
  save_inserted_heatmaps: true
  save_not_inserted_heatmaps: false
  max_inserted_heatmaps_per_sample: 0
  heatmap_layer_window: 4
  save_contact_sheet: true
```

Temporary overrides remain available through `--set`, for example:

```bash
--set trigger_trace.save_not_inserted_heatmaps=true \
      trigger_trace.max_not_inserted_heatmaps_per_sample=3
```

At each real Trigger candidate, the workflow first obtains the Trigger action.
When candidate sink-score collection is enabled, it captures the reasoner's
attention before applying the new latent insertion and computes the current
query's mean attention to the first valid key over the configured layer window.
Heatmap image limits affect only saved images, not which candidates enter the
inserted-versus-not-inserted statistics.

Each heatmap uses the existing MVP visual definition:

- rows are real model layer indices;
- columns are key token positions;
- each cell averages the current query's attention over all heads;
- front and tail key windows use a white `...` gap;
- all images share the fixed `[0, 1]` color range.

The title adds the online decision metadata: generation step, relative
position, current token/delimiter, `P(augment)`, insert/skip action, insertion
rank, final reward, and checkpoint label. A red box marks the first valid key,
a cyan dashed line marks the end of the prompt, and latent-token labels are
colored orange.

Outputs are written as:

```text
<output-dir>/
├── trigger_trace_rows.jsonl
├── trigger_trace_samples.jsonl
├── trigger_sink_score_summary.json
├── trigger_sink_score_comparison.png
└── trigger_trace_heatmaps/
    └── sample_XXXX/
        ├── prompt/
        ├── inference_inserted/
        ├── inference_not_inserted_control/
        └── inference_insertions_contact_sheet.png
```

`trigger_trace_rows.jsonl` records each decision and its pre-insertion
`first_key_attention`; it never contains `sink_mass`. The summary compares only
inference delimiter candidates that were actually evaluated by Trigger while
the insertion budget was still available. Prompt candidates are excluded.

`trigger_sink_score_summary.json` reports pooled count, mean, median, standard
deviation, relative-position statistics, and inserted-minus-not-inserted
differences. It also reports a sample-paired mean difference for samples that
contain both actions. `trigger_sink_score_comparison.png` shows the two score
distributions and score versus relative generation position, because raw
first-key attention can be confounded by context length.

For the three-group comparison, keep `max_candidates_per_sample: 0` in
`configs/run_kodcode_default.yaml`. A positive value is only for debugging and
will truncate the candidate pool before sink top-k selection.

Most parameters should be edited in YAML. For temporary one-off overrides, use
`--set key=value`, for example:

```bash
--set max_new_tokens=512 budget=5 save_candidate_attention_heatmaps=true
```

To save per-candidate attention heatmaps, edit `configs/viz_default.yaml`:

```yaml
save_candidate_attention_heatmaps: true
max_heatmap_candidates_per_sample: 0
heatmap_front_key_count: 32
heatmap_tail_key_count: 160
```

`max_heatmap_candidates_per_sample: 0` saves every delimiter candidate on the
baseline trajectory. Each `*_token_attention.png` keeps the earliest key
positions, omits the long middle if needed, and keeps the latest key positions
near the current candidate. The omitted span is shown as a white `...` column.
The y-axis shows the original model layer indices selected by
`sink_layer_window`, not relative row numbers.

To scan the whole baseline trajectory for strong sink events, edit
`configs/viz_default.yaml`:

```yaml
save_sink_event_heatmaps: true
sink_event_layer_window: 4
sink_event_threshold: 0.2
max_sink_event_heatmaps_per_sample: 0
```

A sink event is triggered when the current query token's average attention to
the first valid key position exceeds the threshold, averaged over the selected
last layers and all heads. Event figures are written under
`attention_heatmaps/<reference_mode>/sample_XXXX/sink_events/`, and
`sink_events.jsonl` records the current token text, step, score, and image path.

## Outputs

The output directory contains:

- `candidate_rows.jsonl`: one row per delimiter candidate, including `sink_mass`,
  `sink_mass_z`, `first_key_attention`, and entropy.
- `sequence_points.jsonl`: one row per baseline generation step, including
  `first_key_attention` and whether the prefix ended with a delimiter.
- `selected_point_rows.jsonl`: one row per selected point branch, including
  group name, score, branch reward, inserted flag, and `U(j)`.
- `group_rows.jsonl`: one row per sample/group. This is the direct source for
  `avg_utility`, selected count, and positive precision.
- `strategy_rows.jsonl`: optional, one row per sample and strategy. `planned_steps`
  are selected from baseline candidates; `inserted_steps` are the positions
  that were actually inserted during rollout.
- `summary.json`: aggregate group metrics under
  `counterfactual_groups_by_reference`.
- `sink_events.jsonl`: optional, one row per strong sink event when
  `save_sink_event_heatmaps` is enabled.
- `trigger_trace_rows.jsonl`: Trigger candidate actions and paths to optional
  pre-insertion heatmaps; no SinkMass-derived fields.
- `trigger_trace_samples.jsonl`: sample reward, completion, actual insertion
  counts, and contact-sheet path for the `trigger_trace` workflow.

## Interpretation

The first-key sink hypothesis is supported if:

```text
avg_utility(candidate_first_key_sink_top5) > avg_utility(candidate_first_k)
avg_utility(sequence_first_key_sink_threshold_0_5) > avg_utility(candidate_first_k)
```

Also inspect `selected_count_mean` and `inserted_count_mean`, because the
threshold group does not have a fixed budget.

Strategy reward is exploratory in this MVP. Do not use it as the main claim
until true online policies are implemented.

`rel_pos` and `pos_bucket` are computed against the actual generated length, not
`max_new_tokens`, so same-bucket random controls the observed generation region.

## Visualization

After an experiment finishes:

```bash
python sinkAwareScript/MVP/visualize_results.py \
  --output-dir output/sink_aware_mvp/qwen2_5_kodcode_sft_posfix_limit20
```

This writes PNG figures and `report.md` under `<output-dir>/figures/`.

To inspect which selected points each group actually used:

```bash
python sinkAwareScript/MVP/analyze_selected_points.py \
  --output-dir output/sink_aware_mvp/qwen2_5_kodcode_sft_posfix_limit20
```

This writes `selected_point_report.md`, `selected_point_summary.json`, and
figures under `<output-dir>/selected_point_figures/`. The most useful figures
are `score_vs_utility_by_group.png`, `step_vs_utility_by_group.png`,
`delimiter_fraction.png`, and `utility_by_group.png`.
