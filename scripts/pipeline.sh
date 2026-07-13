#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/bootstrap.sh"
set -e

# ============================================================================
# MemGen 全流水线脚本：Weaver SFT → Weaver GRPO → Trigger GRPO → 评测
#
# 用法:
#   bash scripts/pipeline.sh
#
# 可选环境变量:
#   DATASET_NAME     - 数据集 (默认 kodcode, 可选 gsm8k/gpqa/kodcode/triviaqa)
#   CUDA_VISIBLE_DEVICES - GPU (默认 0)
#   SKIP_SFT         - 设为 1 跳过 Weaver SFT
#   SKIP_GRPO        - 设为 1 跳过 Weaver GRPO
#   SKIP_TRIGGER     - 设为 1 跳过 Trigger GRPO
#   SKIP_EVAL        - 设为 1 跳过评测
#   LOAD_MODEL_PATH  - 从已有 checkpoint 恢复训练 (覆盖相应阶段的从零开始)
#   WEAVER_INSERTION_STRATEGY - first_k/candidate_sink_threshold/sequence_sink_threshold
#   WEAVER_SINK_SCORE_THRESHOLD - sink 策略阈值 (默认 0.3)
#   WEAVER_SINK_SCORE_LAYER_WINDOW - sink score 使用的最后层数，0 表示全部层
#
# 可选 batch size 覆盖 (均为 per_device 值，总有效 batch = per_device × GPU数 × grad_accum):
#   SFT_BATCH_SIZE          - Weaver SFT per_device batch (默认 1)
#   SFT_GRAD_ACCUM          - Weaver SFT 梯度累积 (默认 1)
#   GRPO_BATCH_SIZE         - Weaver GRPO per_device batch (默认 8)
#   GRPO_GRAD_ACCUM         - Weaver GRPO 梯度累积 (默认 1)
#   TRIGGER_BATCH_SIZE      - Trigger GRPO per_device batch (默认 8)
#   TRIGGER_GRAD_ACCUM      - Trigger GRPO 梯度累积 (默认 4)
#   EVAL_BATCH_SIZE         - 评测 per_device batch (默认 4)
# ============================================================================

# ===== 环境变量 =====
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d '[:space:]')
echo "[pipeline] 使用 ${NUM_GPUS} GPU(s): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export MAIN_PROCESS_PORT=29507

# ===== 模型配置 =====
REASONER_MODEL=${REASONER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
WEAVER_MODEL=${WEAVER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TRIGGER_MODEL=${TRIGGER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}

# ===== 数据集 =====
DATASET_NAME=${DATASET_NAME:-kodcode}
configure_dataset_augmentation_defaults "${DATASET_NAME}"
echo "[pipeline] 数据集: ${DATASET_NAME}"

# ===== 增强参数 =====
# augmentation budget 默认随数据集选择；环境变量可显式覆盖。整个 pipeline
# 只使用这一份值，保证后续阶段加载 checkpoint 时张量形状保持一致。
PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-8}
INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-8}
WEAVER_INSERTION_STRATEGY=${WEAVER_INSERTION_STRATEGY:-first_k}
WEAVER_SINK_SCORE_THRESHOLD=${WEAVER_SINK_SCORE_THRESHOLD:-0.3}
WEAVER_SINK_SCORE_LAYER_WINDOW=${WEAVER_SINK_SCORE_LAYER_WINDOW:-4}

echo "[pipeline] 增强参数: PN=${MAX_PROMPT_AUG_NUM} IN=${MAX_INFERENCE_AUG_NUM} PL=${PROMPT_LATENTS_LEN} IL=${INFERENCE_LATENTS_LEN}"
echo "[pipeline] Weaver 插点策略: ${WEAVER_INSERTION_STRATEGY} threshold=${WEAVER_SINK_SCORE_THRESHOLD} layers=${WEAVER_SINK_SCORE_LAYER_WINDOW}"

# ===== Batch size 配置 =====
# 以下均为 per_device 值 (单卡)，总有效 batch = per_device × GPU数 × grad_accum
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-1}
SFT_GRAD_ACCUM=${SFT_GRAD_ACCUM:-1}
GRPO_BATCH_SIZE=${GRPO_BATCH_SIZE:-8}
GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM:-1}
GRPO_NUM_GENERATIONS=${GRPO_NUM_GENERATIONS:-8}
TRIGGER_BATCH_SIZE=${TRIGGER_BATCH_SIZE:-8}
TRIGGER_GRAD_ACCUM=${TRIGGER_GRAD_ACCUM:-4}
TRIGGER_NUM_GENERATIONS=${TRIGGER_NUM_GENERATIONS:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}

if [ "${SFT_BATCH_SIZE}" -ne 1 ] && [ "${SKIP_SFT:-0}" != "1" ]; then
    echo "[weaver-sft] error: SFT_BATCH_SIZE must be 1 for per-sample augmentation alignment" >&2
    exit 1
fi
if [ "${SKIP_GRPO:-0}" != "1" ]; then
    validate_grpo_grouping "${GRPO_BATCH_SIZE}" "${NUM_GPUS}" "${GRPO_NUM_GENERATIONS}" "weaver-grpo"
fi
if [ "${SKIP_TRIGGER:-0}" != "1" ]; then
    validate_trigger_dataset "${DATASET_NAME}"
    validate_grpo_grouping "${TRIGGER_BATCH_SIZE}" "${NUM_GPUS}" "${TRIGGER_NUM_GENERATIONS}" "trigger-grpo"
fi

echo "[pipeline] Batch 配置:"
echo "  Weaver SFT:   per_device=${SFT_BATCH_SIZE} × ${NUM_GPUS}GPU × ${SFT_GRAD_ACCUM}accum = $((SFT_BATCH_SIZE * NUM_GPUS * SFT_GRAD_ACCUM))"
echo "  Weaver GRPO:  per_device=${GRPO_BATCH_SIZE} × ${NUM_GPUS}GPU × ${GRPO_GRAD_ACCUM}accum = $((GRPO_BATCH_SIZE * NUM_GPUS * GRPO_GRAD_ACCUM))"
echo "  Trigger GRPO: per_device=${TRIGGER_BATCH_SIZE} × ${NUM_GPUS}GPU × ${TRIGGER_GRAD_ACCUM}accum = $((TRIGGER_BATCH_SIZE * NUM_GPUS * TRIGGER_GRAD_ACCUM))"
echo "  Eval:         per_device=${EVAL_BATCH_SIZE} × ${NUM_GPUS}GPU = $((EVAL_BATCH_SIZE * NUM_GPUS))"

# ===== 模型短名 (用于路径匹配) =====
MODEL_SHORT=${REASONER_MODEL##*/}

# ===== 查找最近 checkpoint =====
function find_latest_checkpoint() {
    local dry_run_stage="${1:-stage}"
    if [ "${MEMGEN_DRY_RUN:-0}" = "1" ]; then
        echo ".cache/dry-run/${dry_run_stage}/model"
        return 0
    fi
    # 按时间倒序列出 pn/pl/in/il 参数匹配的所有训练目录下的 model 子目录
    local pattern=".cache/train/${DATASET_NAME}/${MODEL_SHORT}/pn=${MAX_PROMPT_AUG_NUM}_pl=${PROMPT_LATENTS_LEN}_in=${MAX_INFERENCE_AUG_NUM}_il=${INFERENCE_LATENTS_LEN}_*/model"
    local result=$(ls -td ${pattern} 2>/dev/null | head -1)
    echo "${result}"
}

# ===== 阶段开关 =====
SKIP_SFT=${SKIP_SFT:-0}
SKIP_GRPO=${SKIP_GRPO:-0}
SKIP_TRIGGER=${SKIP_TRIGGER:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

if [ "${WEAVER_INSERTION_STRATEGY}" != "first_k" ] && [ "${SKIP_GRPO}" != "1" ]; then
    echo "[pipeline] 错误: sink-aware Weaver 插点目前只支持 SFT，请设置 SKIP_GRPO=1" >&2
    exit 1
fi

# 如果提供 LOAD_MODEL_PATH，则跳过之前的阶段
if [ -n "${LOAD_MODEL_PATH}" ]; then
    echo "[pipeline] 从已有 checkpoint 恢复: ${LOAD_MODEL_PATH}"
    require_checkpoint "${LOAD_MODEL_PATH}" "pipeline"
    sync_checkpoint_latent_lengths "${LOAD_MODEL_PATH}"
    SKIP_SFT=1
    SKIP_GRPO=1
    LOAD_WEAVER_PATH="${LOAD_MODEL_PATH}"
fi

# ===== 记录时间戳 =====
PIPELINE_START=$(date +%s)
echo ""
echo "=============================="
echo "  MemGen Pipeline 开始"
echo "=============================="
echo ""

# ###########################################################################
#  Stage 1: Weaver SFT
# ###########################################################################
if [ "${SKIP_SFT}" = "1" ]; then
    echo "===== Stage 1: Weaver SFT (跳过) ====="
    LOAD_WEAVER_PATH="${LOAD_WEAVER_PATH:-}"
else
    echo ""
    echo "===== Stage 1: Weaver SFT ====="
    STAGE_START=$(date +%s)
    PRE_STAGE_CHECKPOINT=$(find_latest_checkpoint "before-weaver-sft")

    run_accelerate \
        --config_file=configs/zero2.yaml \
        --num_processes=${NUM_GPUS} \
        main.py \
        --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.attn_implementation ${ATTN_IMPLEMENTATION} \
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
        dataset.mode sft \
        run.mode train \
        run.train_weaver True \
        run.train_trigger False \
        run.train_weaver_method sft \
        run.weaver.sft.per_device_train_batch_size ${SFT_BATCH_SIZE} \
        run.weaver.sft.per_device_eval_batch_size ${SFT_BATCH_SIZE} \
        run.weaver.sft.bf16 True \
        run.weaver.sft.gradient_accumulation_steps ${SFT_GRAD_ACCUM}

    LOAD_WEAVER_PATH=$(find_latest_checkpoint "weaver-sft")
    if [ -z "${LOAD_WEAVER_PATH}" ]; then
        echo "[pipeline] 错误: Stage 1 未找到 checkpoint"
        exit 1
    fi
    if [ "${MEMGEN_DRY_RUN:-0}" != "1" ] && [ "${LOAD_WEAVER_PATH}" = "${PRE_STAGE_CHECKPOINT}" ]; then
        echo "[pipeline] 错误: Stage 1 没有产生新的 checkpoint: ${LOAD_WEAVER_PATH}"
        exit 1
    fi

    STAGE_END=$(date +%s)
    echo "[pipeline] Stage 1 耗时: $((STAGE_END - STAGE_START))s"
    echo "[pipeline] Weaver SFT checkpoint: ${LOAD_WEAVER_PATH}"
fi

# ###########################################################################
#  Stage 2: Weaver GRPO
# ###########################################################################
if [ "${SKIP_GRPO}" = "1" ]; then
    echo ""
    echo "===== Stage 2: Weaver GRPO (跳过) ====="
else
    echo ""
    echo "===== Stage 2: Weaver GRPO ====="
    STAGE_START=$(date +%s)

    if [ -z "${LOAD_WEAVER_PATH}" ]; then
        echo "[pipeline] 错误: 未找到 Weaver checkpoint (需要先运行 Stage 1 或设置 LOAD_MODEL_PATH)"
        exit 1
    fi
    require_checkpoint "${LOAD_WEAVER_PATH}" "weaver-grpo"
    echo "[pipeline] 加载 Weaver: ${LOAD_WEAVER_PATH}"
    PRE_STAGE_CHECKPOINT=$(find_latest_checkpoint "before-weaver-grpo")

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
        model.weaver.insertion_strategy.name ${WEAVER_INSERTION_STRATEGY} \
        model.weaver.insertion_strategy.sink_score_threshold ${WEAVER_SINK_SCORE_THRESHOLD} \
        model.weaver.insertion_strategy.sink_score_layer_window ${WEAVER_SINK_SCORE_LAYER_WINDOW} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active False \
        dataset.mode grpo \
        run.mode train \
        run.train_weaver True \
        run.train_trigger False \
        run.train_weaver_method grpo \
        run.weaver.grpo.per_device_train_batch_size ${GRPO_BATCH_SIZE} \
        run.weaver.grpo.per_device_eval_batch_size ${GRPO_BATCH_SIZE} \
        run.weaver.grpo.num_train_epochs 1 \
        run.weaver.grpo.num_generations ${GRPO_NUM_GENERATIONS} \
        run.weaver.grpo.gradient_accumulation_steps ${GRPO_GRAD_ACCUM}

    LOAD_WEAVER_PATH=$(find_latest_checkpoint "weaver-grpo")
    if [ -z "${LOAD_WEAVER_PATH}" ]; then
        echo "[pipeline] 错误: Stage 2 未找到 checkpoint"
        exit 1
    fi
    if [ "${MEMGEN_DRY_RUN:-0}" != "1" ] && [ "${LOAD_WEAVER_PATH}" = "${PRE_STAGE_CHECKPOINT}" ]; then
        echo "[pipeline] 错误: Stage 2 没有产生新的 checkpoint: ${LOAD_WEAVER_PATH}"
        exit 1
    fi

    STAGE_END=$(date +%s)
    echo "[pipeline] Stage 2 耗时: $((STAGE_END - STAGE_START))s"
    echo "[pipeline] Weaver GRPO checkpoint: ${LOAD_WEAVER_PATH}"
fi

# ###########################################################################
#  Stage 3: Trigger GRPO
# ###########################################################################
if [ "${SKIP_TRIGGER}" = "1" ]; then
    echo ""
    echo "===== Stage 3: Trigger GRPO (跳过) ====="
else
    echo ""
    echo "===== Stage 3: Trigger GRPO ====="
    STAGE_START=$(date +%s)

    if [ -z "${LOAD_WEAVER_PATH}" ]; then
        echo "[pipeline] 错误: 未找到 Weaver checkpoint (需要先运行 Stage 1/2 或设置 LOAD_MODEL_PATH)"
        exit 1
    fi
    require_checkpoint "${LOAD_WEAVER_PATH}" "trigger-grpo"
    echo "[pipeline] 加载 Weaver: ${LOAD_WEAVER_PATH}"
    PRE_STAGE_CHECKPOINT=$(find_latest_checkpoint "before-trigger-grpo")

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
        dataset.mode grpo \
        run.mode train \
        run.train_weaver False \
        run.train_trigger True \
        run.train_trigger_method grpo \
        run.trigger.grpo.per_device_train_batch_size ${TRIGGER_BATCH_SIZE} \
        run.trigger.grpo.per_device_eval_batch_size ${TRIGGER_BATCH_SIZE} \
        run.trigger.grpo.num_train_epochs 1 \
        run.trigger.grpo.num_generations ${TRIGGER_NUM_GENERATIONS} \
        run.trigger.grpo.gradient_accumulation_steps ${TRIGGER_GRAD_ACCUM} \
        run.interaction.trigger_do_sample True

    LOAD_TRIGGER_PATH=$(find_latest_checkpoint "trigger-grpo")
    if [ -z "${LOAD_TRIGGER_PATH}" ]; then
        echo "[pipeline] 错误: Stage 3 未找到 checkpoint"
        exit 1
    fi
    if [ "${MEMGEN_DRY_RUN:-0}" != "1" ] && [ "${LOAD_TRIGGER_PATH}" = "${PRE_STAGE_CHECKPOINT}" ]; then
        echo "[pipeline] 错误: Stage 3 没有产生新的 checkpoint: ${LOAD_TRIGGER_PATH}"
        exit 1
    fi
    # 本次 pipeline 刚训练出的 Trigger 必须在随后的评测中启用。
    EVAL_TRIGGER_ACTIVE=True

    STAGE_END=$(date +%s)
    echo "[pipeline] Stage 3 耗时: $((STAGE_END - STAGE_START))s"
    echo "[pipeline] Trigger GRPO checkpoint: ${LOAD_TRIGGER_PATH}"
fi

# ###########################################################################
#  Stage 4: 评测
# ###########################################################################
if [ "${SKIP_EVAL}" = "1" ]; then
    echo ""
    echo "===== Stage 4: 评测 (跳过) ====="
else
    echo ""
    echo "===== Stage 4: 评测 ====="
    STAGE_START=$(date +%s)

    # 优先使用 Trigger checkpoint，否则用 Weaver checkpoint
    EVAL_MODEL_PATH="${LOAD_TRIGGER_PATH:-${LOAD_WEAVER_PATH}}"
    if [ -z "${EVAL_MODEL_PATH}" ]; then
        echo "[pipeline] 错误: 未找到可评测的 checkpoint"
        exit 1
    fi
    require_checkpoint "${EVAL_MODEL_PATH}" "evaluate"
    EVAL_TRIGGER_ACTIVE=${EVAL_TRIGGER_ACTIVE:-${TRIGGER_ACTIVE:-False}}
    echo "[pipeline] 评测模型: ${EVAL_MODEL_PATH}"

    run_accelerate \
        --config_file=configs/zero2.yaml \
        --num_processes=${NUM_GPUS} \
        main.py \
        --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.attn_implementation ${ATTN_IMPLEMENTATION} \
        model.load_model_path ${EVAL_MODEL_PATH} \
        model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        model.weaver.model_name ${WEAVER_MODEL} \
        model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
        model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active ${EVAL_TRIGGER_ACTIVE} \
        run.mode evaluate \
        run.interaction.batch_size ${EVAL_BATCH_SIZE} \
        run.interaction.temperature 0.0 \
        run.interaction.max_response_length 1024

    STAGE_END=$(date +%s)
    echo "[pipeline] Stage 4 耗时: $((STAGE_END - STAGE_START))s"
fi

# ===== 完成 =====
PIPELINE_END=$(date +%s)
echo ""
echo "=============================="
echo "  MemGen Pipeline 完成"
echo "  总耗时: $((PIPELINE_END - PIPELINE_START))s"
echo "=============================="
