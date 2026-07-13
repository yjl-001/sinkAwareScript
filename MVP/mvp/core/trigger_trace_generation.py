from dataclasses import dataclass

import torch

from mvp.core.generation import insert_latent
from mvp.core.model_setup import decode_completion
from mvp.core.records import TriggerTracePointRecord
from mvp.viz.trigger_trace_viz import TriggerHeatmapSnapshot, capture_trigger_heatmap_snapshot


@dataclass
class TriggerTraceGeneration:
    completion_ids: list[int]
    completion: str
    points: list[TriggerTracePointRecord]
    snapshots: list[TriggerHeatmapSnapshot]
    prompt_inserted: bool
    inference_inserted_count: int


@torch.no_grad()
def generate_with_trigger_trace(model, prompt_ids, prompt_mask, *, sample_idx: int, args,
                                trace_config) -> TriggerTraceGeneration:
    """按训练后 Trigger 的真实在线决策生成，并捕获插入前 attention。"""

    if not model.trigger.active:
        raise RuntimeError("trigger_trace requires model.trigger.active=True")

    tokenizer = model.tokenizer
    reasoner = model.reasoner
    current_input_ids = prompt_ids
    current_inputs_embeds = reasoner.get_input_embeddings()(prompt_ids)
    current_attention_mask = prompt_mask
    current_position_ids = model._generate_position_ids(current_attention_mask)
    current_cache = None
    key_labels = initial_key_labels(model, prompt_ids, prompt_mask)
    prompt_len = prompt_ids.size(1)
    inference_insert_count = 0
    prompt_inserted = False
    points: list[TriggerTracePointRecord] = []
    snapshots: list[TriggerHeatmapSnapshot] = []
    inserted_snapshot_count = 0
    skipped_snapshot_count = 0

    for step in range(args.max_new_tokens):
        prefix_ends_with_delimiter = bool(
            step > 0
            and model._check_ends_with_delimiter(current_input_ids, tokenizer, model.delimiters).item()
        )
        is_prompt = step == 0
        is_inference_candidate = (
            prefix_ends_with_delimiter and inference_insert_count < model.config.max_inference_aug_num
        )
        is_candidate = is_prompt or is_inference_candidate

        diagnostic_outputs = None
        if is_candidate:
            action, probability = trigger_action_and_probability(model, current_input_ids, trace_config)
            will_insert = action == 1
            insert_rank = inference_insert_count + 1 if will_insert and not is_prompt else None
            point = build_point(
                model, sample_idx, len(points), step, is_prompt, current_input_ids,
                probability, action, insert_rank, trace_config
            )
            points.append(point)

            capture, inserted_snapshot_count, skipped_snapshot_count = should_capture_snapshot(
                point, trace_config, inserted_snapshot_count, skipped_snapshot_count
            )
            if capture:
                diagnostic_outputs = reasoner_forward(
                    reasoner,
                    current_inputs_embeds,
                    current_attention_mask,
                    current_position_ids,
                    current_cache,
                    output_attentions=True,
                )
                snapshots.append(
                    capture_trigger_heatmap_snapshot(
                        diagnostic_outputs,
                        current_attention_mask,
                        key_labels,
                        prompt_len,
                        point,
                        args,
                        trace_config,
                    )
                )

            if will_insert:
                old_length = current_attention_mask.size(1)
                current_inputs_embeds, current_attention_mask, current_position_ids = insert_latent(
                    model,
                    current_inputs_embeds,
                    current_attention_mask,
                    current_position_ids,
                    is_prompt=is_prompt,
                )
                latent_count = current_attention_mask.size(1) - old_length
                latent_kind = "prompt_latent" if is_prompt else f"inference_latent_{inference_insert_count + 1}"
                key_labels.extend([f"<{latent_kind}:{idx}>" for idx in range(latent_count)])
                current_cache = None
                diagnostic_outputs = None
                if is_prompt:
                    prompt_inserted = True
                else:
                    inference_insert_count += 1

        outputs = diagnostic_outputs
        if outputs is None:
            outputs = reasoner_forward(
                reasoner,
                current_inputs_embeds,
                current_attention_mask,
                current_position_ids,
                current_cache,
                output_attentions=False,
            )
        current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids = model._append_one_step(
            outputs,
            current_inputs_embeds,
            current_attention_mask,
            current_position_ids,
            current_input_ids,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
        current_cache = outputs.past_key_values
        key_labels.append(token_label(model, int(current_input_ids[0, -1].item())))
        if int(current_input_ids[0, -1].item()) == tokenizer.eos_token_id:
            break

    completion_ids = current_input_ids[0, prompt_len:].tolist()
    generated_len = max(len(completion_ids), 1)
    for point in points:
        point.total_generated_len = len(completion_ids)
        point.rel_pos = point.step / generated_len

    return TriggerTraceGeneration(
        completion_ids=completion_ids,
        completion=decode_completion(model, completion_ids),
        points=points,
        snapshots=snapshots,
        prompt_inserted=prompt_inserted,
        inference_inserted_count=inference_insert_count,
    )


def reasoner_forward(reasoner, inputs_embeds, attention_mask, position_ids, cache, *, output_attentions: bool):
    if cache is not None:
        reasoner_inputs = inputs_embeds[:, -1:]
        reasoner_positions = position_ids[:, -1:]
    else:
        reasoner_inputs = inputs_embeds
        reasoner_positions = position_ids
    return reasoner(
        inputs_embeds=reasoner_inputs,
        attention_mask=attention_mask,
        position_ids=reasoner_positions,
        output_attentions=output_attentions,
        output_hidden_states=False,
        use_cache=True,
        past_key_values=cache,
    )


def trigger_action_and_probability(model, input_ids, trace_config) -> tuple[int, float]:
    attention_mask = (input_ids != model.tokenizer.pad_token_id).long()
    position_ids = model._generate_position_ids(attention_mask)
    logits = model.trigger(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )[:, -1]
    temperature = float(trace_config.get("trigger_temperature", 1.0))
    probability_logits = logits.float() / temperature if temperature > 0 else logits.float()
    probability = float(torch.softmax(probability_logits, dim=-1)[0, 1].item())
    action = int(
        model._get_next_token(
            logits,
            do_sample=bool(trace_config.get("trigger_do_sample", False)),
            temperature=temperature,
        )[0, 0].item()
    )
    return action, probability


def build_point(model, sample_idx: int, point_rank: int, step: int, is_prompt: bool,
                current_input_ids, probability: float, action: int, insert_rank: int | None,
                trace_config) -> TriggerTracePointRecord:
    token_id = int(current_input_ids[0, -1].item())
    text = token_label(model, token_id)
    return TriggerTracePointRecord(
        sample_idx=sample_idx,
        point_rank=point_rank,
        step=step,
        point_type="prompt" if is_prompt else "inference",
        generated_so_far=step,
        total_generated_len=0,
        rel_pos=0.0,
        current_token_id=token_id,
        current_token_text=text,
        delimiter_text="<prompt>" if is_prompt else text,
        trigger_probability=probability,
        trigger_action=action,
        actual_inserted=action == 1,
        inference_insert_rank=insert_rank,
        checkpoint_label=str(trace_config.get("checkpoint_label", "trained_trigger")),
    )


def should_capture_snapshot(point, trace_config, inserted_count: int, skipped_count: int):
    if point.point_type == "prompt":
        return bool(trace_config.get("save_prompt_heatmap", False)), inserted_count, skipped_count
    if point.actual_inserted:
        limit = int(trace_config.get("max_inserted_heatmaps_per_sample", 0))
        capture = bool(trace_config.get("save_inserted_heatmaps", True)) and (limit <= 0 or inserted_count < limit)
        return capture, inserted_count + int(capture), skipped_count
    limit = int(trace_config.get("max_not_inserted_heatmaps_per_sample", 5))
    capture = bool(trace_config.get("save_not_inserted_heatmaps", False)) and (limit <= 0 or skipped_count < limit)
    return capture, inserted_count, skipped_count + int(capture)


def initial_key_labels(model, input_ids, attention_mask) -> list[str]:
    labels = []
    for position in range(input_ids.size(1)):
        if int(attention_mask[0, position].item()) == 0:
            labels.append("<pad>")
        else:
            labels.append(token_label(model, int(input_ids[0, position].item())))
    return labels


def token_label(model, token_id: int) -> str:
    text = model.tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\t", "\\t")
