#!/usr/bin/env bash
# 用户常用 KodCode SFT 入口：保留 pn=1/pl=4/in=5/il=4 的实验设置。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATASET_NAME=kodcode
export MAX_PROMPT_AUG_NUM=${MAX_PROMPT_AUG_NUM:-1}
export MAX_INFERENCE_AUG_NUM=${MAX_INFERENCE_AUG_NUM:-5}
export PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-4}
export INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-4}
exec bash "${SCRIPT_DIR}/../weaver_sft.sh"
