from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class GroupConfig:
    """一个可比较实验组的配置。

    source 决定从哪里取点：candidate 只看 delimiter 候选点，sequence 看整条
    baseline 轨迹；selector 决定如何从这些点里挑选待反事实评估的位置。
    """

    name: str
    source: str
    selector: str
    score: str = "first_key_attention"
    budget: int | None = None
    threshold: float | None = None
    requires_delimiter: bool = True


@dataclass
class ExperimentConfig:
    """实验层配置，专门描述 reference 和要比较的点选择方案。"""

    max_prompt_aug_num: int
    first_key_layer_window: int
    groups: list[GroupConfig]

    @property
    def prompt_augment(self) -> bool:
        return self.max_prompt_aug_num > 0

    @property
    def reference_mode(self) -> str:
        return "prompt_only" if self.prompt_augment else "no_memory"


def load_experiment_config(path: str | None) -> ExperimentConfig:
    """读取 YAML 配置，并转成带类型的 dataclass。"""

    if not path:
        path = "sinkAwareScript/MVP/configs/first_key_sink_three_groups.yaml"
    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    groups = [GroupConfig(**item) for item in data.get("groups", [])]
    return ExperimentConfig(
        max_prompt_aug_num=int(data.get("max_prompt_aug_num", 1)),
        first_key_layer_window=int(data.get("first_key_layer_window", 4)),
        groups=groups,
    )
