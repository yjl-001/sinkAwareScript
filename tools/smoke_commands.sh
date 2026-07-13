#!/usr/bin/env bash
# 不加载模型，只检查 shell 参数路由、四数据集默认值和预期失败分支。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

assert_contains() {
    local output="$1"
    local expected="$2"
    if [[ "${output}" != *"${expected}"* ]]; then
        echo "[failed] command output is missing: ${expected}" >&2
        exit 1
    fi
}

assert_not_contains() {
    local output="$1"
    local unexpected="$2"
    if [[ "${output}" == *"${unexpected}"* ]]; then
        echo "[failed] command output unexpectedly contains: ${unexpected}" >&2
        exit 1
    fi
}

for dataset in gsm8k gpqa kodcode triviaqa; do
    sft_output=$(MEMGEN_DRY_RUN=1 DATASET_NAME="${dataset}" bash scripts/weaver_sft.sh)
    assert_contains "${sft_output}" "dataset.mode sft"
    assert_contains "${sft_output}" "run.weaver.sft.per_device_train_batch_size 1"
    assert_contains "${sft_output}" "--num_processes=1"
    assert_not_contains "${sft_output}" "--num_processes= 1"

    grpo_output=$(MEMGEN_DRY_RUN=1 DATASET_NAME="${dataset}" bash scripts/weaver_grpo.sh)
    assert_contains "${grpo_output}" "dataset.mode grpo"
    assert_contains "${grpo_output}" "run.train_weaver True"

    eval_output=$(MEMGEN_DRY_RUN=1 DATASET_NAME="${dataset}" LOAD_MODEL_PATH=/tmp/fake bash scripts/eval.sh)
    assert_contains "${eval_output}" "run.mode evaluate"
done

trivia_output=$(MEMGEN_DRY_RUN=1 DATASET_NAME=triviaqa bash scripts/weaver_sft.sh)
assert_contains "${trivia_output}" "model.max_prompt_aug_num 8"
assert_contains "${trivia_output}" "model.max_inference_aug_num 0"

for dataset in gsm8k gpqa kodcode; do
    trigger_output=$(MEMGEN_DRY_RUN=1 DATASET_NAME="${dataset}" LOAD_WEAVER_PATH=/tmp/fake \
        bash scripts/trigger_train.sh)
    assert_contains "${trigger_output}" "model.trigger.active True"
    assert_contains "${trigger_output}" "run.train_trigger True"
    assert_contains "${trigger_output}" "run.interaction.trigger_do_sample True"
done

# 完整静态 pipeline 与跳过 Trigger 的 TriviaQA pipeline 都必须能完成参数展开。
pipeline_output=$(MEMGEN_DRY_RUN=1 DATASET_NAME=kodcode bash scripts/pipeline.sh)
assert_contains "${pipeline_output}" "model.trigger.active True"
assert_contains "${pipeline_output}" "run.mode evaluate"
MEMGEN_DRY_RUN=1 DATASET_NAME=triviaqa SKIP_TRIGGER=1 bash scripts/pipeline.sh >/dev/null

if MEMGEN_DRY_RUN=1 DATASET_NAME=triviaqa LOAD_WEAVER_PATH=/tmp/fake \
    bash scripts/trigger_train.sh >/dev/null 2>&1; then
    echo "[failed] TriviaQA Trigger GRPO should be rejected" >&2
    exit 1
fi

if MEMGEN_DRY_RUN=1 BATCH_SIZE=2 bash scripts/weaver_sft.sh >/dev/null 2>&1; then
    echo "[failed] Weaver SFT batch_size=2 should be rejected" >&2
    exit 1
fi

if MEMGEN_DRY_RUN=1 GRPO_BATCH_SIZE=3 NUM_GENERATIONS=8 bash scripts/weaver_grpo.sh >/dev/null 2>&1; then
    echo "[failed] invalid GRPO grouping should be rejected" >&2
    exit 1
fi

echo "[ok] command routing smoke tests passed"
