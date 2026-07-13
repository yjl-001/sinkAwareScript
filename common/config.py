import json
import logging
import math
from pathlib import Path

from omegaconf import OmegaConf

class Config:
    def __init__(self, args):
        self.config = {}

        self.args = args

        user_config = self._build_opt_list(self.args.options)

        config = OmegaConf.load(self.args.cfg_path)
        runner_config = self.build_runner_config(config, **user_config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config, **user_config)

        # Override the default configuration with user options.
        self.config = OmegaConf.merge(
            runner_config, model_config, dataset_config, user_config
        )
        self._sync_checkpoint_metadata()
        self._validate()


    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    @staticmethod
    def build_model_config(config, **kwargs):
        return {"model": config.model}

    @staticmethod
    def build_runner_config(config, **kwargs):
        return {"run": config.run}

    @staticmethod
    def build_dataset_config(config, **kwargs):
        dataset = config.get("dataset", None)
        if dataset is None:
            raise KeyError(
                "Expecting 'dataset' as the root key for dataset configuration."
            )

        return dict(dataset=dataset)

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = ["=" in opt for opt in opts]
        if all(has_equal):
            return opts
        if any(has_equal) or len(opts) % 2 != 0:
            raise ValueError(
                "--options must use either key=value items or an even number of key value items"
            )

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def _sync_checkpoint_metadata(self):
        """在创建工作目录前用 checkpoint 元数据校正 latent 长度。"""

        load_model_path = self.config.model.get("load_model_path")
        if not load_model_path:
            return
        config_path = Path(str(load_model_path)).expanduser() / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"MemGen checkpoint is missing config.json: {config_path}")
        checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
        for checkpoint_key, runtime_key in (
            ("prompt_latents_len", "prompt_latents_len"),
            ("inference_latents_len", "inference_latents_len"),
        ):
            if checkpoint_key not in checkpoint_config:
                raise KeyError(f"Checkpoint config is missing {checkpoint_key}: {config_path}")
            checkpoint_value = int(checkpoint_config[checkpoint_key])
            runtime_value = int(self.config.model.weaver[runtime_key])
            if checkpoint_value != runtime_value:
                logging.warning(
                    "Override runtime %s=%s with checkpoint value %s",
                    runtime_key,
                    runtime_value,
                    checkpoint_value,
                )
                self.config.model.weaver[runtime_key] = checkpoint_value

    def get_config(self):
        return self.config

    def _validate(self):
        """在加载模型前检查跨配置不变量，避免错误配置跑数小时后才暴露。"""

        dataset = self.config.dataset
        model = self.config.model
        run = self.config.run
        supported_datasets = {"gsm8k", "kodcode", "gpqa", "triviaqa"}
        if dataset.name not in supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset.name}")
        if dataset.mode not in {"sft", "grpo"}:
            raise ValueError(f"dataset.mode must be sft or grpo, got: {dataset.mode}")
        if run.mode not in {"train", "evaluate"}:
            raise ValueError(f"run.mode must be train or evaluate, got: {run.mode}")
        if int(model.max_prompt_aug_num) < 0 or int(model.max_inference_aug_num) < 0:
            raise ValueError("augmentation budgets must be non-negative")
        if int(model.weaver.prompt_latents_len) <= 0 or int(model.weaver.inference_latents_len) <= 0:
            raise ValueError("latent lengths must be positive")
        if model.get("attn_implementation", "flash_attention_2") not in {
            "flash_attention_2", "sdpa", "eager"
        }:
            raise ValueError("model.attn_implementation must be flash_attention_2, sdpa, or eager")

        insertion_strategy = model.weaver.get("insertion_strategy", {})
        insertion_strategy_name = str(insertion_strategy.get("name", "first_k"))
        supported_insertion_strategies = {
            "first_k", "candidate_sink_threshold", "sequence_sink_threshold"
        }
        if insertion_strategy_name not in supported_insertion_strategies:
            raise ValueError(
                f"Unsupported Weaver insertion strategy: {insertion_strategy_name}"
            )
        sink_threshold = float(insertion_strategy.get("sink_score_threshold", 0.0))
        if not math.isfinite(sink_threshold):
            raise ValueError("model.weaver.insertion_strategy.sink_score_threshold must be finite")
        if int(insertion_strategy.get("sink_score_layer_window", 4)) < 0:
            raise ValueError(
                "model.weaver.insertion_strategy.sink_score_layer_window must be >= 0"
            )

        if run.mode != "train":
            return

        train_weaver = bool(run.train_weaver)
        train_trigger = bool(run.train_trigger)
        if train_weaver == train_trigger:
            raise ValueError("Training must enable exactly one of run.train_weaver/run.train_trigger")

        if train_weaver:
            method = str(run.train_weaver_method)
            if dataset.mode != method:
                raise ValueError(
                    f"Weaver training requires dataset.mode={method}, got {dataset.mode}. "
                    "Use the singular key 'dataset.mode'."
                )
            if bool(model.trigger.active):
                raise ValueError("Weaver training requires model.trigger.active=False")
            if int(model.max_prompt_aug_num) == 0 and int(model.max_inference_aug_num) == 0:
                raise ValueError("Weaver training requires at least one enabled augmentation type")
            if method == "sft" and int(run.weaver.sft.per_device_train_batch_size) != 1:
                raise ValueError(
                    "Weaver SFT requires per_device_train_batch_size=1 because training-time "
                    "augmentation points are selected independently for each sample."
                )
            if method == "grpo" and float(run.weaver.grpo.temperature) <= 0:
                raise ValueError("Weaver GRPO sampling temperature must be positive")
            if method == "grpo" and insertion_strategy_name != "first_k":
                raise NotImplementedError(
                    "Sink-aware Weaver insertion strategies currently support SFT only: "
                    "GRPO rollout generation must use the same insertion policy as "
                    "teacher-forced logprob recomputation."
                )

        if train_trigger:
            if str(run.train_trigger_method) != "grpo" or dataset.mode != "grpo":
                raise ValueError("Trigger training supports GRPO only and requires dataset.mode=grpo")
            if dataset.name == "triviaqa":
                raise NotImplementedError(
                    "Trigger GRPO for TriviaQA is not implemented: multi-turn "
                    "augmentation-mask alignment is required."
                )
            if not bool(model.trigger.active):
                raise ValueError("Trigger training requires model.trigger.active=True")
            if not model.load_model_path:
                raise ValueError("Trigger training requires model.load_model_path for a trained Weaver checkpoint")
            if float(run.trigger.grpo.temperature) <= 0:
                raise ValueError("Trigger GRPO sampling temperature must be positive")

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def dataset_cfg(self):
        return self.config.dataset

    @property
    def model_cfg(self):
        return self.config.model

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.dataset))

        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)
