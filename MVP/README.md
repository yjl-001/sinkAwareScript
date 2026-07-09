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
- `first_key_sink_three_groups.yaml`: prompt augmentation and group selectors.
- `viz_default.yaml`: heatmap and sink-event visualization settings.

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
