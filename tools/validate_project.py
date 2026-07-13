#!/usr/bin/env python3
"""不加载模型的数据/训练/实验结构验证。

这个脚本只使用 Python 标准库，适合在没有 torch/GPU 的开发机上执行。它验证
容易静默漂移的跨文件契约；真正的数值训练仍需在服务器做最小样本 smoke test。
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("gsm8k", "gpqa", "kodcode", "triviaqa")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_python_syntax() -> None:
    for path in ROOT.rglob("*.py"):
        if any(part in {".git", ".cache", "__pycache__"} for part in path.parts):
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def validate_shell_syntax() -> None:
    scripts = sorted((ROOT / "scripts").rglob("*.sh"))
    result = subprocess.run(
        ["bash", "-n", *map(str, scripts)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    require(result.returncode == 0, result.stderr or "bash -n failed")


def validate_dataset_closure() -> None:
    registry = (ROOT / "data/__init__.py").read_text(encoding="utf-8")
    for name in DATASETS:
        config = ROOT / f"configs/latent_memory/{name}.yaml"
        require(config.is_file(), f"missing dataset config: {config}")
        text = config.read_text(encoding="utf-8")
        top_keys = {
            line.split(":", 1)[0]
            for line in text.splitlines()
            if line and not line[0].isspace() and ":" in line
        }
        require({"model", "dataset", "run"} <= top_keys, f"invalid config roots: {config}")
        require(f'"{name}"' in registry, f"dataset is not registered: {name}")


def validate_training_contracts() -> None:
    all_shell = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "scripts").rglob("*.sh"))
    require("datasets.mode" not in all_shell, "found invalid plural config key: datasets.mode")

    bootstrap = (ROOT / "scripts/bootstrap.sh").read_text(encoding="utf-8")
    require("python -m accelerate.commands.launch" in bootstrap, "run_accelerate does not launch Accelerate")
    require('MEMGEN_RUN_ID="${run_id}"' in bootstrap, "distributed ranks do not share a run id")

    config = (ROOT / "common/config.py").read_text(encoding="utf-8")
    require("per_device_train_batch_size=1" in config, "missing Weaver SFT batch alignment guard")
    require("Trigger GRPO for TriviaQA is not implemented" in config, "missing TriviaQA Trigger guard")

    weaver_trainer = (ROOT / "memgen/trainer/weaver_grpo_trainer.py").read_text(encoding="utf-8")
    require("batch_size = batch_size or 1" in weaver_trainer, "Weaver GRPO logprob is not per trajectory")
    require("labels_batch = labels[start : start + batch_size]" in weaver_trainer, "labels are not chunk-aligned")

    trigger_trainer = (ROOT / "memgen/trainer/trigger_grpo_trainer.py").read_text(encoding="utf-8")
    require("self.generation_config.trigger_do_sample = True" in trigger_trainer, "Trigger sampling is disabled")
    require("self.generation_config.weaver_do_sample = False" in trigger_trainer, "Weaver sampling must be fixed")

    runner = (ROOT / "memgen/runner.py").read_text(encoding="utf-8")
    require("self.model.open_component('weaver')" in runner, "Weaver target is not reopened after checkpoint load")
    require("self.model.open_component('trigger')" in runner, "Trigger target is not reopened after checkpoint load")

    lora_switch = (ROOT / "memgen/model/modeling_utils.py").read_text(encoding="utf-8")
    require("fix_model_parameters(component.model)" in lora_switch, "PEFT base model is not frozen")
    require("p.requires_grad = True" in lora_switch, "target LoRA parameters are not reopened")

    generation = (ROOT / "memgen/model/modeling_utils.py").read_text(encoding="utf-8")
    require("token_text.rstrip(\" \\t\").endswith" in generation, "missing merged-token delimiter fallback")
    require("prompt_candidate_mask" in generation, "prompt augmentation budget is not enforced")
    require("delimiter attention_mask must match input_ids" in generation, "delimiter path guesses padding from token ids")

    multiturn = (ROOT / "interactions/multiturn_interaction.py").read_text(encoding="utf-8")
    require("prompt_augmentation_counts" in multiturn, "multi-turn prompt budget is not tracked")

    model = (ROOT / "memgen/model/modeling_memgen.py").read_text(encoding="utf-8")
    require("MemGenConfig.from_pretrained(load_model_path)" in model, "checkpoint config is not authoritative")
    require("finished_mask" in model, "batched generation does not isolate finished sequences")


def validate_mvp_paths() -> None:
    configs = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "MVP/configs").glob("*.yaml"))
    require("sinkAwareScript/MVP" not in configs, "MVP config still depends on the parent repository path")
    repo_paths = (ROOT / "MVP/mvp/core/repo_paths.py").read_text(encoding="utf-8")
    require('PROJECT_ROOT = Path(__file__).resolve().parents[3]' in repo_paths, "MVP root points outside repository")
    loader = (ROOT / "MVP/mvp/core/model_setup.py").read_text(encoding="utf-8")
    require("MemGenConfig.from_pretrained(load_model_path)" in loader, "MVP ignores checkpoint latent lengths")
    trace = (ROOT / "MVP/mvp/core/trigger_trace_generation.py").read_text(encoding="utf-8")
    require("current_token_attention_mask" in trace, "MVP Trigger trace uses token-id padding guesses")


def validate_checkpoint_helpers() -> None:
    """用最小假 checkpoint 验证 shell 元数据同步，不依赖 torch。"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint = Path(tmp_dir)
        (checkpoint / "config.json").write_text(
            '{"prompt_latents_len": 4, "inference_latents_len": 6}',
            encoding="utf-8",
        )
        for filename in ("projs.bin", "weaver.bin", "trigger.bin"):
            (checkpoint / filename).touch()
        for adapter_name in ("weaver", "trigger"):
            adapter_dir = checkpoint / adapter_name / adapter_name
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").touch()
        command = (
            'source scripts/bootstrap.sh; '
            f'require_checkpoint "{checkpoint}" test; '
            'PROMPT_LATENTS_LEN=8; INFERENCE_LATENTS_LEN=8; '
            f'sync_checkpoint_latent_lengths "{checkpoint}"; '
            'test "$PROMPT_LATENTS_LEN" = 4; test "$INFERENCE_LATENTS_LEN" = 6'
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        require(result.returncode == 0, result.stderr or "checkpoint metadata sync failed")


def main() -> int:
    checks = (
        validate_python_syntax,
        validate_shell_syntax,
        validate_dataset_closure,
        validate_training_contracts,
        validate_mvp_paths,
        validate_checkpoint_helpers,
    )
    for check in checks:
        check()
        print(f"[ok] {check.__name__}")
    print(f"[ok] project contracts validated: {ROOT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        raise SystemExit(1)
