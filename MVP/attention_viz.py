from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def maybe_save_candidate_heatmaps(candidate, model, current_input_ids, current_attention_mask,
                                  outputs, sample_idx: int, step: int, prompt_len: int, args) -> None:
    """保存当前 baseline candidate 的 token attention 热力图。"""

    if not args.save_candidate_attention_heatmaps:
        return
    # max_heatmap_candidates_per_sample<=0 表示保存所有 candidate；这适合本轮
    # 逐点检查 baseline 轨迹。调试超长样本时可以设成正数，只看前几个点。
    if 0 < args.max_heatmap_candidates_per_sample <= candidate.candidate_rank:
        return
    out_dir = Path(args.output_dir) / "attention_heatmaps" / candidate.reference_mode / f"sample_{sample_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"cand_{candidate.candidate_rank:03d}_step_{step}"
    save_token_attention_heatmap(
        candidate, model, current_input_ids, current_attention_mask, outputs, prompt_len, args, prefix
    )


def save_token_attention_heatmap(candidate, model, current_input_ids, current_attention_mask,
                                 outputs, prompt_len: int, args, prefix: Path) -> None:
    """画当前 candidate query 对 key positions 的 attention。

    为了突出 sink/front tokens，同时避免长序列图片过宽，x 轴保留：
    1. 最前面的 heatmap_front_key_count 个 key；
    2. 靠近当前 candidate 的最后 heatmap_tail_key_count 个 key；
    3. 如果中间被省略，用一个空白分隔列标成 "..."。

    heat: [num_selected_layers, L_key_visible (+ optional gap column)]
    """

    # selected_layers: list[(layer_index, layer_attention)]，layer_index 是模型
    # 原始层号；例如 28 层模型取最后 4 层时是 24,25,26,27。
    selected_layers = select_layers(outputs.attentions, args.sink_layer_window)
    valid_positions = torch.nonzero(current_attention_mask[0] != 0, as_tuple=False).flatten()
    visible_positions, gap_after_front = choose_visible_positions(
        valid_positions, args.heatmap_front_key_count, args.heatmap_tail_key_count
    )
    if visible_positions.numel() == 0:
        return

    rows = []
    layer_indices = []
    for layer_index, layer_attn in selected_layers:
        layer_indices.append(layer_index)
        # query_to_keys: [num_heads, L_key]，当前 candidate 的最后一个 query
        # 对所有历史 key position 的 attention。
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

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.18), max(3, len(rows) * 0.35)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    image = ax.imshow(heat, aspect="auto", cmap=cmap)
    ax.set_title(
        f"sample={candidate.sample_idx} mode={candidate.reference_mode} "
        f"candidate={candidate.candidate_rank} step={candidate.step} "
        f"delimiter={candidate.delimiter_text!r}"
    )
    ax.set_xlabel("key token position")
    ax.set_ylabel("model layer index")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([str(index) for index in layer_indices])
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(f"{prefix}_token_attention.png", dpi=180)
    plt.close(fig)


def choose_visible_positions(valid_positions, front_count: int, tail_count: int):
    """选择 x 轴要展示的 key positions。

    返回值:
    - visible_positions: 真实 key position 序列；
    - gap_after_front: 是否在前段和后段之间插入 "..." 空白列。
    """

    front_count = max(front_count, 0)
    tail_count = max(tail_count, 0)
    total = valid_positions.numel()
    if total <= front_count + tail_count or front_count == 0 or tail_count == 0:
        return valid_positions, False
    front = valid_positions[:front_count]
    tail = valid_positions[-tail_count:]
    return torch.cat([front, tail]), True


def add_gap_column(row: torch.Tensor, front_count: int) -> torch.Tensor:
    """在前段和后段之间插入 NaN，imshow 会把它画成白色分隔列。"""

    gap = torch.tensor([float("nan")], dtype=row.dtype)
    return torch.cat([row[:front_count], gap, row[front_count:]], dim=0)


def select_layers(attentions, sink_layer_window: int):
    """返回被选中的真实模型层号和对应 attention。

    sink_layer_window=0 表示全部层；否则取最后 N 层。这里保留真实层号，
    避免热力图 y 轴只显示 0,1,2 这种相对行号。
    """

    total_layers = len(attentions)
    start = max(0, total_layers - sink_layer_window) if sink_layer_window > 0 else 0
    return list(enumerate(attentions[start:], start=start))


def token_labels(model, current_input_ids, positions, prompt_len: int, latent_count: int,
                 gap_after_front: bool, args) -> list[str]:
    labels = []
    token_len = current_input_ids.size(1)
    for pos in positions.tolist():
        token_pos = real_token_position(pos, prompt_len, latent_count)
        if token_pos is not None and token_pos < token_len:
            text = model.tokenizer.decode([int(current_input_ids[0, token_pos].item())], skip_special_tokens=False)
            text = text.replace("\n", "\\n").replace("\t", "\\t")
            labels.append(f"{pos}:{text[:12]}")
        else:
            labels.append(f"{pos}:<latent>")
    if gap_after_front:
        labels.insert(args.heatmap_front_key_count, "...")
    return labels


def real_token_position(position: int, prompt_len: int, latent_count: int) -> int | None:
    """把 augmented key position 映射回 current_input_ids 里的真实 token 位置。"""

    if latent_count == 0 or position < prompt_len:
        return position
    if position < prompt_len + latent_count:
        return None
    return position - latent_count
