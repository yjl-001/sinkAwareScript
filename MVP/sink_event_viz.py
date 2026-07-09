import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from attention_viz import (
    add_gap_column,
    choose_visible_positions,
    first_key_attention_score,
    select_layers,
    token_labels,
)


def maybe_save_sink_event_heatmap(model, current_input_ids, current_attention_mask,
                                  outputs, sample_idx: int, step: int, prompt_len: int, args) -> None:
    """扫描 baseline 轨迹中的强 sink 事件，并按需保存 attention 热力图。

    这里的 sink event 定义为：当前 query token 对第一个有效 key position 的
    attention，在最后 N 层和所有 heads 上取平均后超过阈值。
    """

    if not args.save_sink_event_heatmaps or step == 0:
        return
    score = first_key_attention_score(outputs.attentions, current_attention_mask, args.sink_event_layer_window)
    if not torch.isfinite(torch.tensor(score)) or score < args.sink_event_threshold:
        return

    event_rank = next_event_rank(args, sample_idx)
    if 0 < args.max_sink_event_heatmaps_per_sample <= event_rank:
        return

    out_dir = Path(args.output_dir) / "attention_heatmaps" / args.current_reference_mode
    out_dir = out_dir / f"sample_{sample_idx:04d}" / "sink_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"sink_event_{event_rank:03d}_step_{step}"
    image_path = save_sink_event_token_heatmap(
        model, current_input_ids, current_attention_mask, outputs,
        sample_idx, step, prompt_len, score, event_rank, args, prefix
    )
    append_sink_event_row(model, current_input_ids, sample_idx, step, prompt_len, score, event_rank, image_path, args)


def save_sink_event_token_heatmap(model, current_input_ids, current_attention_mask, outputs,
                                  sample_idx: int, step: int, prompt_len: int, score: float,
                                  event_rank: int, args, prefix: Path) -> str:
    """保存强 sink 事件这一刻的 current-query attention 热力图。"""

    selected_layers = select_layers(outputs.attentions, args.sink_event_layer_window)
    valid_positions = torch.nonzero(current_attention_mask[0] != 0, as_tuple=False).flatten()
    visible_positions, gap_after_front = choose_visible_positions(
        valid_positions, args.heatmap_front_key_count, args.heatmap_tail_key_count
    )
    rows, layer_indices = [], []
    for layer_index, layer_attn in selected_layers:
        layer_indices.append(layer_index)
        query_to_keys = layer_attn[0, :, -1, :].float()
        usable = visible_positions[visible_positions < query_to_keys.size(-1)]
        row = query_to_keys[:, usable].mean(dim=0).cpu()
        if gap_after_front:
            row = add_gap_column(row, args.heatmap_front_key_count)
        rows.append(row)

    heat = torch.stack(rows).numpy()
    latent_count = max(0, current_attention_mask.size(1) - current_input_ids.size(1))
    labels = token_labels(
        model, current_input_ids, visible_positions, prompt_len, latent_count, gap_after_front, args
    )
    current_text = current_token_text(model, current_input_ids)
    completion_index = current_input_ids.size(1) - prompt_len - 1

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.18), max(3, len(rows) * 0.35)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    image = ax.imshow(heat, aspect="auto", cmap=cmap)
    ax.set_title(
        f"sink event sample={sample_idx} mode={args.current_reference_mode} "
        f"event={event_rank} step={step} score={score:.4f}\n"
        f"current token[{completion_index}]={short_text(current_text, 90)!r}"
    )
    ax.set_xlabel("key token position")
    ax.set_ylabel("model layer index")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([str(index) for index in layer_indices])
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    image_path = f"{prefix}_token_attention.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    return image_path


def append_sink_event_row(model, current_input_ids, sample_idx: int, step: int, prompt_len: int,
                          score: float, event_rank: int, image_path: str, args) -> None:
    """把强 sink 事件写成 jsonl，方便后续直接统计 token 内容。"""

    row = {
        "sample_idx": sample_idx,
        "reference_mode": args.current_reference_mode,
        "event_rank": event_rank,
        "step": step,
        "completion_token_index": current_input_ids.size(1) - prompt_len - 1,
        "current_token_text": current_token_text(model, current_input_ids),
        "first_key_attention": score,
        "layer_window": args.sink_event_layer_window,
        "threshold": args.sink_event_threshold,
        "image_path": image_path,
    }
    with (Path(args.output_dir) / "sink_events.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def next_event_rank(args, sample_idx: int) -> int:
    """返回当前 sample 的 sink event 序号，并更新计数。"""

    counts = getattr(args, "_sink_event_counts", {})
    key = (args.current_reference_mode, sample_idx)
    rank = counts.get(key, 0)
    counts[key] = rank + 1
    setattr(args, "_sink_event_counts", counts)
    return rank


def current_token_text(model, current_input_ids) -> str:
    """解码当前 query 对应的最后一个真实 token。"""

    token_id = int(current_input_ids[0, -1].item())
    text = model.tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\t", "\\t")


def short_text(text: str, max_len: int) -> str:
    """限制标题里的 token 文本长度，防止 Matplotlib 标题过宽。"""

    return text if len(text) <= max_len else text[: max_len - 3] + "..."
