import argparse
from datetime import datetime
import os
from pathlib import Path
import random

import numpy as np
import torch

from common.config import Config
from common.logger import setup_logger
from data import get_data_builder
from memgen.model import MemGenModel
from memgen.runner import MemGenRunner


# 训练输出、相对配置路径和本地包都以独立仓库根目录为准。这样无论调用者
# 当前位于哪个目录，`python /path/to/main.py` 都不会误用上层 MemGen 文件。
PROJECT_ROOT = Path(__file__).resolve().parent

def set_seed(random_seed: int, use_gpu: bool):
    """固定 Python / NumPy / PyTorch 的随机源，方便复现实验结果。"""
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"set seed: {random_seed}")

def parse_args():
    """只解析配置文件路径和 OmegaConf dotlist 覆盖项。

    绝大多数训练/评测参数都留在 YAML 中；命令行只负责选择配置文件，
    以及用 `--options key value` 的形式覆盖少量实验变量。
    """
    parser = argparse.ArgumentParser(description="Memory Generator")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args

def build_working_dir(config: Config) -> str:
    """根据运行模式、数据集、模型和 latent 配置生成本次实验目录。

    目录名把 prompt/inference 两类 latent 的数量和长度写进去，
    这样后续看 `.cache/` 时可以直接从路径判断实验条件。
    """

    # parent dir: <train/evaluate>/<dataset_name>/<reasoner_model_name>
    mode = config.run_cfg.mode
    dataset_name = config.dataset_cfg.name
    model_name = config.model_cfg.model_name.rstrip("/").split("/")[-1]
    parent_dir = PROJECT_ROOT / ".cache" / mode / dataset_name / model_name

    # name: <prompt_aug_num>_<prompt_latents_len>_<inference_aug_num>_<inference_latents_len>_<timestamp>
    max_prompt_aug_num = config.model_cfg.max_prompt_aug_num
    prompt_latents_len = config.model_cfg.weaver.prompt_latents_len
    max_inference_aug_num = config.model_cfg.max_inference_aug_num
    inference_latents_len = config.model_cfg.weaver.inference_latents_len
    # Accelerate 的所有 rank 继承 launcher 生成的同一个 run id，避免各进程
    # 在秒边界上算出不同目录。直接运行 main.py 时仍回退到本地时间。
    time = os.environ.get("MEMGEN_RUN_ID", datetime.now().strftime("%Y%m%d-%H%M%S"))
    working_dir = f"pn={max_prompt_aug_num}_pl={prompt_latents_len}_in={max_inference_aug_num}_il={inference_latents_len}_{time}"

    return str(parent_dir / working_dir)

def main():
    """命令行入口：配置 -> 数据 -> 模型 -> Runner -> train/evaluate。"""

    # 保证 YAML 中的相对路径、评测输出和 shell 脚本采用同一个路径语义。
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    cfg_path = Path(args.cfg_path)
    if not cfg_path.is_absolute():
        args.cfg_path = str(PROJECT_ROOT / cfg_path)
    config = Config(args)

    set_seed(config.run_cfg.seed, use_gpu=torch.cuda.is_available())

    # set up working directory
    working_dir = build_working_dir(config)

    # set up logger
    config.run_cfg.log_dir = os.path.join(working_dir, "logs")
    setup_logger(output_dir=config.run_cfg.log_dir)

    config.pretty_print()

    # build components
    # Config.to_dict() 后会被拆成 dataset/model/run 三块：
    # - data_builder 负责构造 HuggingFace Dataset 和 Env 类
    # - MemGenModel.from_config 负责加载 reasoner/weaver/trigger 三份底座模型
    # - Runner 负责把训练器、交互循环和评测 recorder 串起来
    config_dict = config.to_dict()
    data_builder = get_data_builder(config_dict.get("dataset"))
    model = MemGenModel.from_config(config_dict.get("model"))

    runner = MemGenRunner(
        model=model,
        data_builder=data_builder,
        config=config_dict,
        working_dir=working_dir
    )

    # train or evaluate
    if config.run_cfg.mode == "train":
        runner.train()

    elif config.run_cfg.mode == "evaluate":
        runner.evaluate()

if __name__ == "__main__":
    main()
