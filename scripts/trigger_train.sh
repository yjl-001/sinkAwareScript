#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/bootstrap.sh"
set -e

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d '[:space:]')
echo "Using $NUM_GPUS GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL=${REASONER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
WEAVER_MODEL=${WEAVER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TRIGGER_MODEL=${TRIGGER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}

# Dataset configs
DATASET_NAME=${DATASET_NAME:-kodcode}  # Trigger GRPO currently supports static datasets.
configure_dataset_augmentation_defaults "${DATASET_NAME}"
validate_trigger_dataset "${DATASET_NAME}"

# MemGen configs
TRAIN_METHOD="grpo"

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-8}
INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-8}


LOAD_WEAVER_PATH=${LOAD_WEAVER_PATH:-}
TRIGGER_BATCH_SIZE=${TRIGGER_BATCH_SIZE:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

require_checkpoint "${LOAD_WEAVER_PATH}" "trigger-grpo"
validate_grpo_grouping "${TRIGGER_BATCH_SIZE}" "${NUM_GPUS}" "${NUM_GENERATIONS}" "trigger-grpo"

# train
run_accelerate \
    --config_file=configs/zero2.yaml \
    --num_processes=${NUM_GPUS} \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.attn_implementation ${ATTN_IMPLEMENTATION} \
    model.load_model_path ${LOAD_WEAVER_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active True \
    dataset.mode ${TRAIN_METHOD} \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method ${TRAIN_METHOD} \
    run.trigger.grpo.per_device_train_batch_size ${TRIGGER_BATCH_SIZE} \
    run.trigger.grpo.per_device_eval_batch_size ${TRIGGER_BATCH_SIZE} \
    run.trigger.grpo.num_train_epochs 1 \
    run.trigger.grpo.num_generations ${NUM_GENERATIONS} \
    run.trigger.grpo.gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    run.interaction.trigger_do_sample True \
