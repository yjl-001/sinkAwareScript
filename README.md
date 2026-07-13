# Sink-Aware MemGen

本仓库是从本地 MemGen 基线迁入的独立训练、评测与 sink-aware 实验工程。训练主链路
位于仓库根目录，`MVP/` 作为可复用实验层，不再依赖上层 `/Users/.../MemGen` 中的代码。

## 支持范围

| 数据集 | Weaver SFT | Weaver GRPO | Trigger GRPO | 评测 | MVP 反事实/热力图 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 是 | 是 | 是 | 是 | 是，单轮 |
| GPQA | 是 | 是 | 是 | 是 | 是，单轮 |
| KodCode | 是 | 是 | 是 | 是 | 是，单轮 |
| TriviaQA | 是 | 是 | 否 | 是 | 否，多轮 |

TriviaQA 使用 `DynamicEnv`。它的 Weaver GRPO 与评测依赖 `dataset.retrieval.search_url`
指定的检索服务。当前 Trigger Trainer 还没有把逐轮 augmentation mask 对齐到最终多轮
history，因此配置层会明确拒绝 TriviaQA Trigger GRPO，避免训练错误 action。

## 环境

```bash
conda create -n memgen python=3.10
conda activate memgen
pip install -r requirements.txt
```

主配置默认使用 `flash_attention_2`，服务器需已有 FlashAttention。没有该依赖时可在命令
前设置 `ATTN_IMPLEMENTATION=sdpa`；sink heatmap 实验会单独使用 `eager` 读取 attention。

默认使用 `configs/zero2.yaml` 的 DeepSpeed ZeRO-2，只支持 DDP。脚本会按
`CUDA_VISIBLE_DEVICES` 自动计算进程数，并为同一次启动的所有 rank 注入相同 run id，
确保日志和 checkpoint 写入同一工作目录。

## 规范入口

四个入口都可通过环境变量切换 `DATASET_NAME=gsm8k|gpqa|kodcode|triviaqa`。脚本会从
自身位置定位仓库根目录，所以不要求调用者先 `cd` 到本目录。

Weaver SFT：

```bash
CUDA_VISIBLE_DEVICES=0,1 DATASET_NAME=kodcode \
PROMPT_LATENTS_LEN=4 INFERENCE_LATENTS_LEN=4 \
bash scripts/weaver_sft.sh
```

Weaver SFT 的 `per_device_train_batch_size` 必须为 1。当前训练期插点选择是逐样本逻辑，
配置和脚本都会阻止更大的本卡 batch；多卡仍可正常使用，每卡各处理一条样本。

Weaver GRPO：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DATASET_NAME=kodcode \
LOAD_MODEL_PATH=/path/to/weaver-sft/model \
GRPO_BATCH_SIZE=8 NUM_GENERATIONS=8 \
bash scripts/weaver_grpo.sh
```

Trigger GRPO，仅 GSM8K/GPQA/KodCode：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DATASET_NAME=kodcode \
LOAD_WEAVER_PATH=/path/to/trained-weaver/model \
TRIGGER_BATCH_SIZE=8 NUM_GENERATIONS=8 \
bash scripts/trigger_train.sh
```

Trigger rollout 固定 `weaver_do_sample=False`，开启 `trigger_do_sample=True`。GRPO 的
`per_device_batch * GPU 数` 必须能被 `num_generations` 整除，脚本会在加载模型前检查。

评测：

```bash
DATASET_NAME=kodcode LOAD_MODEL_PATH=/path/to/checkpoint/model \
TRIGGER_ACTIVE=False bash scripts/eval.sh  # 只评 Weaver

DATASET_NAME=kodcode LOAD_MODEL_PATH=/path/to/trigger/model \
TRIGGER_ACTIVE=True bash scripts/eval.sh   # 评 Weaver + Trigger
```

checkpoint 必须是 `save_pretrained` 生成的 `model/` 目录。恢复时 latent 长度与 LoRA
结构以 checkpoint 的 `config.json` 为准；CLI 中的 augmentation budget 和
`trigger.active` 仍作为运行时配置覆盖。

完整 pipeline：

```bash
DATASET_NAME=kodcode CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/pipeline.sh
DATASET_NAME=triviaqa SKIP_TRIGGER=1 bash scripts/pipeline.sh
```

`scripts/train/` 与 `scripts/eval/` 下的文件是历史实验条件的薄包装器，真实参数路由统一
复用以上四个规范入口。评测 checkpoint 不再写死在这些包装器中。

## 配置与输出

- `configs/latent_memory/*.yaml`：四套数据、模型、SFT/GRPO 和 interaction 配置。
- `model.max_prompt_aug_num` / `model.max_inference_aug_num`：两类插入预算。
- `model.weaver.*_latents_len`：新训练时的 latent 长度。
- `run.interaction.*_do_sample`：评测/交互采样开关。
- `.cache/train|evaluate/...`：按数据集、模型和 latent 条件生成的运行目录。

默认 augmentation budget：GSM8K/GPQA/KodCode 为 `prompt=1, inference=5`；TriviaQA
为 `prompt=8, inference=0`。环境变量可覆盖，但同一 pipeline 的所有阶段必须一致。

## Sink-Aware 实验

`MVP/` 保留 candidate 反事实、Trigger 在线轨迹、first-key attention 热力图、插入/未插入
候选对比及离线重绘。通用静态任务入口为：

```bash
python MVP/run_sink_experiment.py \
  --run-config MVP/configs/run_kodcode_trigger_trace.yaml \
  --load-model-path /path/to/trained-trigger/model \
  --output-dir output/sink_aware_mvp/kodcode_trigger_trace \
  --limit 20 \
  --overwrite
```

KodCode 旧入口 `MVP/run_kodcode_sink_mvp.py` 继续兼容。MVP 从 checkpoint 配置读取 latent
长度，因此不会再因运行 YAML 的 `8` 与 checkpoint 的 `4` 不一致而 shape mismatch。

## 验证

开发机上不加载模型的逻辑验证：

```bash
python tools/validate_project.py
bash tools/smoke_commands.sh
```

`MEMGEN_DRY_RUN=1` 也可用于查看任一训练脚本展开后的完整 Accelerate 命令。服务器上还应
先用极小数据/步数分别做一次 SFT、Weaver GRPO、Trigger GRPO 和评测 smoke run，并核对
trainable 参数、非零梯度、augmentation mask 以及保存后重载结果。
