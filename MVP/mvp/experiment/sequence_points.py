from mvp.core.records import SequencePointRecord
from mvp.metrics.sink_metrics import first_key_attention_score


def maybe_add_sequence_point(points, model, current_input_ids, current_attention_mask,
                             outputs, sample_idx: int, step: int,
                             prefix_ends_with_delimiter: bool, args) -> None:
    """记录 baseline 轨迹上的一个普通 step。

    step=0 对应 prompt 后第一个 token 生成前的位置，语义上更接近 prompt
    augmentation，不纳入“inference latent 插入点”的全序列扫描。
    """

    if step == 0:
        return
    token_id = int(current_input_ids[0, -1].item())
    text = model.tokenizer.decode([token_id], skip_special_tokens=False)
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    rel_pos = step / max(args.max_new_tokens, 1)
    points.append(
        SequencePointRecord(
            sample_idx=sample_idx,
            reference_mode=getattr(args, "current_reference_mode", "unknown"),
            point_rank=len(points),
            step=step,
            generated_so_far=step,
            rel_pos=rel_pos,
            pos_bucket=min(int(rel_pos * 4), 3),
            token_id=token_id,
            token_text=text,
            prefix_ends_with_delimiter=bool(prefix_ends_with_delimiter),
            first_key_attention=first_key_attention_score(
                outputs.attentions,
                current_attention_mask,
                args.first_key_layer_window,
            ),
        )
    )
