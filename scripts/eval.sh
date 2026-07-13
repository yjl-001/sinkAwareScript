#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/bootstrap.sh"
set -e

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAIN_PROCESS_PORT=29508

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
TRIGGER_ACTIVE=${TRIGGER_ACTIVE:-False}


# Dataset configs
DATASET_NAME=${DATASET_NAME:-kodcode}  # gsm8k, gpqa, kodcode, triviaqa
configure_dataset_augmentation_defaults "${DATASET_NAME}"

# MemGen configs

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=8, MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-8}
INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-8}

BATCH_SIZE=${BATCH_SIZE:-4}

# Trained model path:
# - Must point to a checkpoint directory containing projs.bin/weaver.bin/trigger.bin.
# - Required when evaluating the model
LOAD_MODEL_PATH=${LOAD_MODEL_PATH:-}

require_checkpoint "${LOAD_MODEL_PATH}" "evaluate"

# evaluate
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
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active ${TRIGGER_ACTIVE} \
    run.mode evaluate \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 1024 \
