import argparse
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf

from mvp.core.repo_paths import resolve_project_path


DEFAULT_RUN_CONFIG = "MVP/configs/run_kodcode_default.yaml"


def parse_args() -> argparse.Namespace:
    """读取运行配置，并允许命令行覆盖少量高频字段。

    大部分实验参数放在 YAML 里，避免 CLI 变成“参数垃圾桶”。命令行只保留
    配置文件路径、输出目录、样本范围这类经常临时改的字段。
    """

    parser = argparse.ArgumentParser(description="Run a sink-aware latent-memory experiment.")
    parser.add_argument("--run-config", default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--viz-config", default=None)
    parser.add_argument("--trigger-trace-config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--load-model-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--set", nargs="*", default=None, help="Override run/viz config with key=value pairs.")
    parser.add_argument("--options", nargs="+", default=None, help="OmegaConf dotlist overrides for MemGen config.")
    cli = parser.parse_args()

    run_config_path = resolve_project_path(cli.run_config, must_exist=True)
    run_config = load_yaml(run_config_path)
    viz_path = cli.viz_config or run_config.get("viz_config")
    trigger_trace_path = cli.trigger_trace_config or run_config.get("trigger_trace_config")
    merged = OmegaConf.merge(
        run_config,
        load_yaml(viz_path) if viz_path else {},
        load_yaml(trigger_trace_path) if trigger_trace_path else {},
    )

    for key in ["experiment_config", "output_dir", "load_model_path", "limit", "start_index", "overwrite"]:
        value = getattr(cli, key)
        if value is not None:
            merged[key] = value
    if cli.set:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(cli.set))
    if cli.options is not None:
        merged["options"] = cli.options

    args = argparse.Namespace(**OmegaConf.to_container(merged, resolve=True))
    # 入口允许从任意 cwd 调用，后续模块只消费绝对路径。
    args.run_config = str(run_config_path)
    args.cfg_path = str(resolve_project_path(args.cfg_path, must_exist=True))
    args.output_dir = str(resolve_project_path(args.output_dir))
    if args.load_model_path:
        args.load_model_path = str(resolve_project_path(args.load_model_path, must_exist=True))
    if getattr(args, "experiment_config", None):
        args.experiment_config = str(resolve_project_path(args.experiment_config, must_exist=True))
    args.viz_config = str(resolve_project_path(viz_path, must_exist=True)) if viz_path else None
    args.trigger_trace_config = (
        str(resolve_project_path(trigger_trace_path, must_exist=True)) if trigger_trace_path else None
    )
    args.device = resolve_device(getattr(args, "device", "auto"))
    return args


def load_yaml(path: str | Path | None):
    """读取 YAML；空路径返回空配置，便于可选配置合并。"""

    if not path:
        return OmegaConf.create({})
    return OmegaConf.load(resolve_project_path(path, must_exist=True))


def resolve_device(device: str) -> str:
    """支持配置里写 auto，让脚本在本地/服务器都能自然选择设备。"""

    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """固定随机源，让 random/sampling 策略可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    """把配置字符串转成 torch dtype。"""

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def apply_overrides(config, options: list[str] | None):
    """支持和 main.py 一样的 OmegaConf dotlist 覆盖风格。"""

    if not options:
        return config
    has_equal = ["=" in option for option in options]
    if all(has_equal):
        override = OmegaConf.from_dotlist(options)
    else:
        if any(has_equal) or len(options) % 2 != 0:
            raise ValueError(
                "--options must use either key=value items or an even number of key value items"
            )
        pairs = [f"{key}={value}" for key, value in zip(options[0::2], options[1::2])]
        override = OmegaConf.from_dotlist(pairs)
    return OmegaConf.merge(config, override)


def to_plain_dict(node) -> dict:
    """把 OmegaConf 节点转为普通 dict，方便传给项目已有 builder/model。"""

    return OmegaConf.to_container(node, resolve=True)
