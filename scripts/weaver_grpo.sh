#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/bootstrap.sh"
set -e

export DEBUG_MODE=false
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAIN_PROCESS_PORT=29508

# 自动计算 GPU 数量
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d '[:space:]')
echo "Using $NUM_GPUS GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL=${REASONER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
WEAVER_MODEL=${WEAVER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TRIGGER_MODEL=${TRIGGER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}

# Dataset configs
DATASET_NAME=${DATASET_NAME:-kodcode}  # options: gsm8k, gpqa, kodcode, triviaqa
configure_dataset_augmentation_defaults "${DATASET_NAME}"

# MemGen configs
TRAIN_METHOD="grpo"    # options: sft or grpo

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-8}
INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-8}
# first_k / candidate_sink_threshold / sequence_sink_threshold
WEAVER_INSERTION_STRATEGY=${WEAVER_INSERTION_STRATEGY:-first_k}
WEAVER_SINK_SCORE_THRESHOLD=${WEAVER_SINK_SCORE_THRESHOLD:-0.3}
WEAVER_SINK_SCORE_LAYER_WINDOW=${WEAVER_SINK_SCORE_LAYER_WINDOW:-4}
if [ "${WEAVER_INSERTION_STRATEGY}" != "first_k" ]; then
    echo "[weaver-grpo] error: sink-aware insertion currently supports Weaver SFT only" >&2
    exit 1
fi

GRPO_BATCH_SIZE=${GRPO_BATCH_SIZE:-${GROUP_SIZE:-8}}
NUM_GENERATIONS=${NUM_GENERATIONS:-${GROUP_SIZE:-8}}

LOAD_MODEL_PATH=${LOAD_MODEL_PATH:-null}
if [ "${LOAD_MODEL_PATH}" != "null" ]; then
    require_checkpoint "${LOAD_MODEL_PATH}" "weaver-grpo"
fi

validate_grpo_grouping "${GRPO_BATCH_SIZE}" "${NUM_GPUS}" "${NUM_GENERATIONS}" "weaver-grpo"

# train
run_accelerate \
    --config_file=configs/zero2.yaml \
    --num_processes=${NUM_GPUS} \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.attn_implementation ${ATTN_IMPLEMENTATION} \
    model.load_model_path ${LOAD_MODEL_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.weaver.insertion_strategy.name ${WEAVER_INSERTION_STRATEGY} \
    model.weaver.insertion_strategy.sink_score_threshold ${WEAVER_SINK_SCORE_THRESHOLD} \
    model.weaver.insertion_strategy.sink_score_layer_window ${WEAVER_SINK_SCORE_LAYER_WINDOW} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active False \
    dataset.mode ${TRAIN_METHOD} \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method ${TRAIN_METHOD} \
    run.weaver.grpo.max_completion_length 512 \
    run.weaver.grpo.num_train_epochs 1 \
    run.weaver.grpo.per_device_train_batch_size ${GRPO_BATCH_SIZE} \
    run.weaver.grpo.per_device_eval_batch_size ${GRPO_BATCH_SIZE} \
    run.weaver.grpo.num_generations ${NUM_GENERATIONS} \
    run.weaver.grpo.gradient_accumulation_steps 1 \
