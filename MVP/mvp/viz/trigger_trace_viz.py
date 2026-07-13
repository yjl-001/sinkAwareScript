from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from mvp.core.records import TriggerTracePointRecord
from mvp.viz.attention_viz import add_gap_column, choose_visible_positions, select_layers


@dataclass
class TriggerHeatmapSnapshot:
    """一次 Trigger 决策前的 attention heatmap 快照。"""

    point: TriggerTracePointRecord
    heat: torch.Tensor
    layer_indices: list[int]
    labels: list[str]
    display_positions: list[int | None]
    first_key_position: int
    prompt_end_position: int


def capture_trigger_heatmap_snapshot(outputs, attention_mask, key_labels: list[str], prompt_len: int,
                                     point: TriggerTracePointRecord, args, trace_config) -> TriggerHeatmapSnapshot:
    """按现有 MVP 定义捕获 current-query 对 key positions 的逐层 attention。"""

    selected_layers = select_layers(outputs.attentions, int(trace_config.get("heatmap_layer_window", 4)))
    valid_positions = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
    visible_positions, gap_after_front = choose_visible_positions(
        valid_positions,
        int(args.heatmap_front_key_count),
        int(args.heatmap_tail_key_count),
    )
    if visible_positions.numel() == 0:
        raise RuntimeError("Trigger heatmap has no visible key positions")

    rows, layer_indices = [], []
    for layer_index, layer_attn in selected_layers:
        layer_indices.append(layer_index)
        query_to_keys = layer_attn[0, :, -1, :].float()
        usable = visible_positions[visible_positions < query_to_keys.size(-1)]
        row = query_to_keys[:, usable].mean(dim=0).cpu()
        if gap_after_front:
            row = add_gap_column(row, int(args.heatmap_front_key_count))
        rows.append(row)

    positions: list[int | None] = visible_positions.tolist()
    labels = [label_for_position(key_labels, pos) for pos in visible_positions.tolist()]
    if gap_after_front:
        gap_index = int(args.heatmap_front_key_count)
        positions.insert(gap_index, None)
        labels.insert(gap_index, "...")

    return TriggerHeatmapSnapshot(
        point=point,
        heat=torch.stack(rows),
        layer_indices=layer_indices,
        labels=labels,
        display_positions=positions,
        first_key_position=int(valid_positions[0].item()),
        prompt_end_position=prompt_len - 1,
    )


def save_trigger_trace_visuals(snapshots: list[TriggerHeatmapSnapshot], sample_record, args,
                               trace_config) -> str | None:
    """保存逐点图和按生成顺序排列的实际插入 contact sheet。"""

    inserted_paths = []
    for snapshot in snapshots:
        path = trigger_heatmap_path(snapshot.point, args)
        save_trigger_heatmap(snapshot, path)
        snapshot.point.image_path = str(path)
        if snapshot.point.point_type == "inference" and snapshot.point.actual_inserted:
            inserted_paths.append(path)

    if not bool(trace_config.get("save_contact_sheet", True)) or not inserted_paths:
        return None
    contact_path = Path(args.output_dir) / "trigger_trace_heatmaps"
    contact_path = contact_path / f"sample_{sample_record.sample_idx:04d}" / "inference_insertions_contact_sheet.png"
    save_contact_sheet(
        inserted_paths,
        contact_path,
        columns=max(1, int(trace_config.get("contact_sheet_columns", 2))),
        sample_idx=sample_record.sample_idx,
    )
    return str(contact_path)


def save_trigger_heatmap(snapshot: TriggerHeatmapSnapshot, path: Path) -> None:
    """渲染单个 Trigger 决策前的 heatmap，并附加决策与轨迹信息。"""

    point = snapshot.point
    heat = snapshot.heat.numpy()
    fig, ax = plt.subplots(figsize=(max(9, len(snapshot.labels) * 0.18), max(3.8, len(snapshot.layer_indices) * 0.4)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    image = ax.imshow(heat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    mark_key_position(ax, snapshot, snapshot.first_key_position, color="#e41a1c", linewidth=1.8)
    mark_prompt_boundary(ax, snapshot)
    action_text = "INSERT" if point.actual_inserted else "NO_INSERT"
    insert_rank = "-" if point.inference_insert_rank is None else str(point.inference_insert_rank)
    reward_text = "?" if point.reward is None else f"{point.reward:.4f}"
    ax.set_title(
        f"PRE-INSERTION ATTENTION | sample={point.sample_idx} step={point.step} "
        f"rel_pos={point.rel_pos:.3f} type={point.point_type}\n"
        f"trigger={action_text} p(augment)={point.trigger_probability:.4f} "
        f"inference_insert={insert_rank} reward={reward_text} checkpoint={point.checkpoint_label}\n"
        f"current_token={point.current_token_text!r} delimiter={point.delimiter_text!r}"
    )
    ax.set_xlabel("key token position (red box = first valid key; cyan line = prompt end)")
    ax.set_ylabel("model layer index")
    ax.set_xticks(range(len(snapshot.labels)))
    ax.set_xticklabels(snapshot.labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(snapshot.layer_indices)))
    ax.set_yticklabels([str(index) for index in snapshot.layer_indices])
    color_latent_labels(ax)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def trigger_heatmap_path(point: TriggerTracePointRecord, args) -> Path:
    if point.point_type == "prompt":
        category = "prompt"
    elif point.actual_inserted:
        category = "inference_inserted"
    else:
        category = "inference_not_inserted_control"
    directory = Path(args.output_dir) / "trigger_trace_heatmaps" / f"sample_{point.sample_idx:04d}" / category
    name = f"point_{point.point_rank:03d}_step_{point.step}_{'insert' if point.actual_inserted else 'skip'}.png"
    return directory / name


def save_contact_sheet(paths: list[Path], output_path: Path, *, columns: int, sample_idx: int) -> None:
    rows = (len(paths) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 8, rows * 5.2), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, path in zip(axes.flat, paths):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.stem, fontsize=9)
    fig.suptitle(f"sample={sample_idx} trained Trigger inference insertions", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def label_for_position(key_labels: list[str], position: int) -> str:
    label = key_labels[position] if position < len(key_labels) else "<unknown>"
    return f"{position}:{label[:18]}"


def mark_key_position(ax, snapshot: TriggerHeatmapSnapshot, position: int, *, color: str, linewidth: float) -> None:
    if position not in snapshot.display_positions:
        return
    index = snapshot.display_positions.index(position)
    ax.add_patch(
        Rectangle(
            (index - 0.5, -0.5),
            1,
            len(snapshot.layer_indices),
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
    )


def mark_prompt_boundary(ax, snapshot: TriggerHeatmapSnapshot) -> None:
    if snapshot.prompt_end_position not in snapshot.display_positions:
        return
    index = snapshot.display_positions.index(snapshot.prompt_end_position)
    ax.axvline(index + 0.5, color="#00bcd4", linewidth=1.2, linestyle="--")


def color_latent_labels(ax) -> None:
    for label in ax.get_xticklabels():
        if "latent" in label.get_text():
            label.set_color("#d95f02")
