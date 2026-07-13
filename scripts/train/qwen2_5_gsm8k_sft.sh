#!/usr/bin/env bash
# 历史实验入口：固定 Qwen2.5/GSM8K，保留原实验的 inference budget=3。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATASET_NAME=gsm8k
export MAX_PROMPT_AUG_NUM=${MAX_PROMPT_AUG_NUM:-1}
export MAX_INFERENCE_AUG_NUM=${MAX_INFERENCE_AUG_NUM:-3}
export PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-8}
export INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-8}
exec bash "${SCRIPT_DIR}/../weaver_sft.sh"
