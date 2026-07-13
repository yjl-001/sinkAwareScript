#!/usr/bin/env bash
# SmolLM3/KodCode 评测入口。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REASONER_MODEL=${REASONER_MODEL:-HuggingFaceTB/SmolLM3-3B}
export WEAVER_MODEL=${WEAVER_MODEL:-HuggingFaceTB/SmolLM3-3B}
export DATASET_NAME=kodcode
export PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-4}
export INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-4}
exec bash "${SCRIPT_DIR}/../eval.sh"
