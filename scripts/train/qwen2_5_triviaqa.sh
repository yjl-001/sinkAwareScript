#!/usr/bin/env bash
# Qwen2.5/TriviaQA Weaver SFT 入口；TriviaQA 使用多轮 prompt augmentation。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATASET_NAME=triviaqa
export MAX_PROMPT_AUG_NUM=${MAX_PROMPT_AUG_NUM:-8}
export MAX_INFERENCE_AUG_NUM=${MAX_INFERENCE_AUG_NUM:-0}
export PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-8}
export INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-8}
exec bash "${SCRIPT_DIR}/../weaver_sft.sh"
