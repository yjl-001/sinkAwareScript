from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def maybe_save_candidate_heatmaps(candidate, model, current_input_ids, current_attention_mask,
                                  outputs, sample_idx: int, step: int, args) -> None:
    """保存当前 candidate 的 token attention 和 layer-head sink mass 热力图。"""

    if not args.save_candidate_attention_heatmaps:
        return
    if candidate.candidate_rank >= args.max_heatmap_candidates_per_sample:
        return
    out_dir = Path(args.output_dir) / "attention_heatmaps" / candidate.reference_mode / f"sample_{sample_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"cand_{candidate.candidate_rank:03d}_step_{step}"
    save_token_attention_heatmap(model, current_input_ids, current_attention_mask, outputs, args, prefix)
    save_layer_head_sink_heatmap(current_attention_mask, outputs, args, prefix)


def save_token_attention_heatmap(model, current_input_ids, current_attention_mask, outputs, args, prefix: Path) -> None:
    """画 selected layers 聚合后的 query-to-key attention。

    heat: [num_selected_layers, L_key_visible]
    """

    layers = select_layers(outputs.attentions, args.sink_layer_window)
    valid_positions = torch.nonzero(current_attention_mask[0] != 0, as_tuple=False).flatten()
    visible_positions = valid_positions[-args.heatmap_key_limit:]
    if visible_positions.numel() == 0:
        return

    rows = []
    for layer_attn in layers:
        query_to_keys = layer_attn[0, :, -1, :].float()
        usable = visible_positions[visible_positions < query_to_keys.size(-1)]
        rows.append(query_to_keys[:, usable].mean(dim=0).cpu())
    heat = torch.stack(rows).numpy()
    labels = token_labels(model, current_input_ids, visible_positions)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.18), max(3, len(rows) * 0.35)))
    image = ax.imshow(heat, aspect="auto", cmap="viridis")
    ax.set_title("Current-query attention to visible keys")
    ax.set_xlabel("key token position")
    ax.set_ylabel("selected layer")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(i) for i in range(len(rows))])
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(f"{prefix}_token_attention.png", dpi=180)
    plt.close(fig)


def save_layer_head_sink_heatmap(current_attention_mask, outputs, args, prefix: Path) -> None:
    """画每层每个 head 对 sink keys 的 attention mass。

    heat: [num_selected_layers, num_heads]
    """

    valid_positions = torch.nonzero(current_attention_mask[0] != 0, as_tuple=False).flatten()
    sink_positions = valid_positions[:args.sink_key_count]
    if sink_positions.numel() == 0:
        return

    rows = []
    for layer_attn in select_layers(outputs.attentions, args.sink_layer_window):
        query_to_keys = layer_attn[0, :, -1, :].float()
        usable = sink_positions[sink_positions < query_to_keys.size(-1)]
        rows.append(query_to_keys[:, usable].sum(dim=-1).cpu())
    heat = torch.stack(rows).numpy()

    fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.35)))
    image = ax.imshow(heat, aspect="auto", cmap="magma")
    ax.set_title("Layer-head sink mass")
    ax.set_xlabel("head")
    ax.set_ylabel("selected layer")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(f"{prefix}_head_sink_mass.png", dpi=180)
    plt.close(fig)


def select_layers(attentions, sink_layer_window: int):
    return attentions[-sink_layer_window:] if sink_layer_window > 0 else attentions


def token_labels(model, current_input_ids, positions) -> list[str]:
    labels = []
    token_len = current_input_ids.size(1)
    for pos in positions.tolist():
        if pos < token_len:
            text = model.tokenizer.decode([int(current_input_ids[0, pos].item())], skip_special_tokens=False)
            text = text.replace("\n", "\\n").replace("\t", "\\t")
            labels.append(f"{pos}:{text[:12]}")
        else:
            labels.append(f"{pos}:<latent>")
    return labels
