#!/usr/bin/env bash
# Qwen2.5/GSM8K SFT 条件评测入口；checkpoint 由 LOAD_MODEL_PATH 提供。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATASET_NAME=gsm8k
export MAX_PROMPT_AUG_NUM=${MAX_PROMPT_AUG_NUM:-1}
export MAX_INFERENCE_AUG_NUM=${MAX_INFERENCE_AUG_NUM:-3}
export PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-8}
export INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-8}
exec bash "${SCRIPT_DIR}/../eval.sh"
