#!/usr/bin/env bash

# 所有训练/评测脚本都会 source 本文件。以脚本位置定位仓库根目录，避免调用者
# 的 cwd 决定 Python 导入、配置文件和 .cache 输出落到哪里。
MEMGEN_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMGEN_REPO_ROOT
cd "${MEMGEN_REPO_ROOT}"

# MEMGEN_DRY_RUN=1 时只打印 accelerate 命令，用于在无 GPU/无 torch 的机器上
# 验证参数路由，不加载数据集和模型。
run_accelerate() {
    if [ "${MEMGEN_DRY_RUN:-0}" = "1" ]; then
        printf '[dry-run] run_accelerate'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    # run id 在父 shell 生成一次，所有 Accelerate rank 继承同一值；加入 PID 和
    # RANDOM 可区分同一秒内启动的多个阶段。
    local run_id
    run_id="${MEMGEN_RUN_ID_OVERRIDE:-$(date +%Y%m%d-%H%M%S)-$$-${RANDOM}}"
    MEMGEN_RUN_ID="${run_id}" python -m accelerate.commands.launch "$@"
}

# 四套数据流程共享同一组数据集级默认值；调用者显式设置环境变量时保留覆盖值。
configure_dataset_augmentation_defaults() {
    local dataset_name="$1"
    local default_prompt_aug
    local default_inference_aug

    case "${dataset_name}" in
        gsm8k|gpqa|kodcode)
            default_prompt_aug=1
            default_inference_aug=5
            ;;
        triviaqa)
            default_prompt_aug=8
            default_inference_aug=0
            ;;
        *)
            echo "[config] error: unsupported dataset: ${dataset_name}" >&2
            return 1
            ;;
    esac

    MAX_PROMPT_AUG_NUM="${MAX_PROMPT_AUG_NUM:-${default_prompt_aug}}"
    MAX_INFERENCE_AUG_NUM="${MAX_INFERENCE_AUG_NUM:-${default_inference_aug}}"
}

validate_trigger_dataset() {
    local dataset_name="$1"
    if [ "${dataset_name}" = "triviaqa" ]; then
        echo "[trigger-grpo] error: TriviaQA is multi-turn; Trigger action-mask alignment is not implemented." >&2
        return 1
    fi
}

require_checkpoint() {
    local checkpoint="$1"
    local stage="$2"
    if [ -z "${checkpoint}" ]; then
        echo "[${stage}] error: checkpoint path is required" >&2
        return 1
    fi
    if [ "${MEMGEN_DRY_RUN:-0}" != "1" ] && [ ! -d "${checkpoint}" ]; then
        echo "[${stage}] error: checkpoint directory does not exist: ${checkpoint}" >&2
        return 1
    fi
    if [ "${MEMGEN_DRY_RUN:-0}" != "1" ]; then
        local required_path
        for required_path in config.json projs.bin weaver.bin trigger.bin; do
            if [ ! -e "${checkpoint}/${required_path}" ]; then
                echo "[${stage}] error: checkpoint is missing ${required_path}: ${checkpoint}" >&2
                return 1
            fi
        done
        local adapter_name
        local adapter_dir
        for adapter_name in weaver trigger; do
            adapter_dir="${checkpoint}/${adapter_name}/${adapter_name}"
            if [ ! -f "${adapter_dir}/adapter_config.json" ]; then
                echo "[${stage}] error: checkpoint is missing ${adapter_name} adapter config: ${adapter_dir}" >&2
                return 1
            fi
            if [ ! -f "${adapter_dir}/adapter_model.safetensors" ] && \
               [ ! -f "${adapter_dir}/adapter_model.bin" ]; then
                echo "[${stage}] error: checkpoint is missing ${adapter_name} adapter weights: ${adapter_dir}" >&2
                return 1
            fi
        done
    fi
}

sync_checkpoint_latent_lengths() {
    local checkpoint="$1"
    if [ "${MEMGEN_DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    local lengths
    lengths=$(python -c 'import json, sys; c=json.load(open(sys.argv[1])); print(c["prompt_latents_len"], c["inference_latents_len"])' "${checkpoint}/config.json")
    local checkpoint_prompt_len
    local checkpoint_inference_len
    read -r checkpoint_prompt_len checkpoint_inference_len <<< "${lengths}"
    if [ "${PROMPT_LATENTS_LEN}" != "${checkpoint_prompt_len}" ] || \
       [ "${INFERENCE_LATENTS_LEN}" != "${checkpoint_inference_len}" ]; then
        echo "[checkpoint] latent lengths overridden by config.json: " \
             "PL=${checkpoint_prompt_len}, IL=${checkpoint_inference_len}"
    fi
    PROMPT_LATENTS_LEN="${checkpoint_prompt_len}"
    INFERENCE_LATENTS_LEN="${checkpoint_inference_len}"
}

validate_grpo_grouping() {
    local per_device_batch="$1"
    local num_processes="$2"
    local num_generations="$3"
    local stage="$4"
    if [ "${num_generations}" -lt 2 ]; then
        echo "[${stage}] error: num_generations must be at least 2 for group-relative advantages" >&2
        return 1
    fi
    local global_micro_batch=$((per_device_batch * num_processes))
    if [ $((global_micro_batch % num_generations)) -ne 0 ]; then
        echo "[${stage}] error: per_device_batch * num_processes (${global_micro_batch}) " \
             "must be divisible by num_generations (${num_generations})" >&2
        return 1
    fi
}
