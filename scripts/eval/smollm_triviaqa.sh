#!/usr/bin/env bash
# SmolLM3 reasoner + Qwen Weaver 的 TriviaQA 历史评测入口。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REASONER_MODEL=${REASONER_MODEL:-HuggingFaceTB/SmolLM3-3B}
export DATASET_NAME=triviaqa
export PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-4}
export INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-4}
exec bash "${SCRIPT_DIR}/../eval.sh"
