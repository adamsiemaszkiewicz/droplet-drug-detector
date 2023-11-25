# -*- coding: utf-8 -*-

import datetime
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import yaml

from src.aml.components.sample_component.config import ClassificationConfig
from src.aml.components.sample_component.data import ClassificationDataConfig
from src.common.consts.directories import CONFIGS_DIR, ROOT_DIR
from src.common.consts.extensions import YAML
from src.common.consts.project import PROJECT_NAME
from src.common.utils.dtype_converters import str_to_bool, str_to_dict, str_to_float, str_to_int
from src.machine_learning.augmentations.config import AugmentationsConfig
from src.machine_learning.callbacks.config import CallbacksConfig
from src.machine_learning.classification.loss_functions.config import ClassificationLossFunctionConfig
from src.machine_learning.classification.metrics.config import ClassificationMetricsConfig
from src.machine_learning.classification.models.config import ClassificationModelConfig
from src.machine_learning.loggers.config import LoggersConfig
from src.machine_learning.optimizer.config import OptimizerConfig
from src.machine_learning.preprocessing.config import PreprocessingConfig
from src.machine_learning.scheduler.config import SchedulerConfig
from src.machine_learning.trainer.config import TrainerConfig


def load_defaults() -> Dict[str, Any]:
    """
    Loads the default configuration from the default.yaml file.
    """
    with open(CONFIGS_DIR / f"default{YAML}", "r") as file:
        return yaml.safe_load(file)


def rel_paths_to_abs_path(path: str) -> Path:
    """
    Converts a relative path to an absolute path relative to the root directory.
    """
    return ROOT_DIR / path


def create_arg_parser() -> ArgumentParser:
    """
    Creates an argument parser for SampleConfig class.

    Returns:
        ArgumentParser: The argument parser with configuration options.
    """
    parser = ArgumentParser(description="Parse configuration")

    # Defaults
    defaults = load_defaults()

    data_defaults: Dict[str, Any] = defaults.get("data", None)
    preprocessing_defaults: Dict[str, Any] = defaults.get("preprocessing", None)
    model_defaults: Dict[str, Any] = defaults.get("model", None)
    loss_function_defaults: Dict[str, Any] = defaults.get("loss_function", None)
    optimizer_defaults: Dict[str, Any] = defaults.get("optimizer", None)
    scheduler_defaults: Dict[str, Any] = defaults.get("scheduler", None)
    metrics_defaults: Dict[str, Any] = defaults.get("metrics", None)
    augmentations_defaults: Dict[str, Any] = defaults.get("augmentations", None)
    callbacks_defaults: Dict[str, Any] = defaults.get("callbacks", None)
    callbacks_early_stopping_defaults: Dict[str, Any] = callbacks_defaults.get("early_stopping", None)
    callbacks_model_checkpoint_defaults: Dict[str, Any] = callbacks_defaults.get("model_checkpoint", None)
    callbacks_learning_rate_monitor_defaults: Dict[str, Any] = callbacks_defaults.get("learning_rate_monitor", None)
    trainer_defaults: Dict[str, Any] = defaults.get("trainer", None)

    # Directories
    parser.add_argument("--dataset_dir", type=str, default=rel_paths_to_abs_path(data_defaults["dataset_dir"]))
    parser.add_argument("--artifacts_dir", type=str, default=rel_paths_to_abs_path(data_defaults["artifacts_dir"]))

    # Data
    parser.add_argument("--val_split", type=str, default=data_defaults["val_split"])
    parser.add_argument("--test_split", type=str, default=data_defaults["test_split"])
    parser.add_argument("--batch_size", type=str, default=data_defaults["batch_size"])

    # Preprocessing
    for i in range(5):
        transform_default = (
            preprocessing_defaults["name_list"][i] if i < len(preprocessing_defaults["name_list"]) else None
        )
        transform_kwargs_default = (
            preprocessing_defaults["extra_arguments_list"][i]
            if i < len(preprocessing_defaults["extra_arguments_list"])
            else "{}"
        )

        parser.add_argument(f"--preprocessing_transform_{i + 1}", type=str, default=transform_default)
        parser.add_argument(f"--preprocessing_transform_{i + 1}_kwargs", type=str, default=transform_kwargs_default)

    # Model
    parser.add_argument("--model_name", type=str, default=model_defaults["name"])
    parser.add_argument("--pretrained", type=str, default=model_defaults["pretrained"])
    parser.add_argument("--num_classes", type=str, default=model_defaults["num_classes"])
    parser.add_argument("--in_channels", type=str, default=model_defaults["in_channels"])

    # Loss function
    parser.add_argument("--loss_function_name", type=str, default=loss_function_defaults["name"])
    parser.add_argument("--loss_function_kwargs", type=str, default=loss_function_defaults["extra_arguments"])

    # Optimizer
    parser.add_argument("--optimizer_name", type=str, default=optimizer_defaults["name"])
    parser.add_argument("--learning_rate", type=str, default=optimizer_defaults["learning_rate"])
    parser.add_argument("--weight_decay", type=str, default=optimizer_defaults["weight_decay"])
    parser.add_argument("--optimizer_kwargs", type=str, default=optimizer_defaults["extra_arguments"])

    # Scheduler
    parser.add_argument(
        "--scheduler_name", type=str, default=scheduler_defaults["name"] if scheduler_defaults else None
    )
    parser.add_argument(
        "--scheduler_kwargs", type=str, default=scheduler_defaults["extra_arguments"] if scheduler_defaults else {}
    )

    # Metrics
    parser.add_argument("--task_type", type=str, default=metrics_defaults["task"])
    for i in range(5):
        metric_default = metrics_defaults["name_list"][i] if i < len(metrics_defaults["name_list"]) else None
        metric_kwargs_default = (
            metrics_defaults["extra_arguments_list"][i] if i < len(metrics_defaults["extra_arguments_list"]) else "{}"
        )

        parser.add_argument(f"--metric_name_{i + 1}", type=str, default=metric_default)
        parser.add_argument(f"--metric_name_{i + 1}_kwargs", type=str, default=metric_kwargs_default)

    # Augmentations
    for i in range(5):
        augmentation_default = (
            augmentations_defaults["name_list"][i]
            if augmentations_defaults and i < len(augmentations_defaults["name_list"])
            else None
        )
        augmentation_kwargs_default = (
            augmentations_defaults["extra_arguments_list"][i]
            if augmentations_defaults and i < len(augmentations_defaults["extra_arguments_list"])
            else "{}"
        )

        parser.add_argument(f"--augmentation_name_{i + 1}", type=str, default=augmentation_default)
        parser.add_argument(f"--augmentation_name_{i + 1}_kwargs", type=str, default=augmentation_kwargs_default)

    # Callbacks
    parser.add_argument(
        "--callbacks_early_stopping_monitor", type=str, default=callbacks_early_stopping_defaults["monitor"]
    )
    parser.add_argument("--callbacks_early_stopping_mode", type=str, default=callbacks_early_stopping_defaults["mode"])
    parser.add_argument(
        "--callbacks_early_stopping_patience", type=str, default=callbacks_early_stopping_defaults["patience"]
    )
    parser.add_argument(
        "--callbacks_early_stopping_min_delta", type=str, default=callbacks_early_stopping_defaults["min_delta"]
    )
    parser.add_argument(
        "--callbacks_early_stopping_verbose", type=str, default=callbacks_early_stopping_defaults["verbose"]
    )

    parser.add_argument(
        "--callbacks_model_checkpoint_monitor", type=str, default=callbacks_model_checkpoint_defaults["monitor"]
    )
    parser.add_argument(
        "--callbacks_model_checkpoint_mode", type=str, default=callbacks_model_checkpoint_defaults["mode"]
    )
    parser.add_argument(
        "--callbacks_model_checkpoint_save_top_k", type=str, default=callbacks_model_checkpoint_defaults["save_top_k"]
    )
    parser.add_argument(
        "--callbacks_model_checkpoint_filename", type=str, default=callbacks_model_checkpoint_defaults["filename"]
    )
    parser.add_argument(
        "--callbacks_model_checkpoint_verbose", type=str, default=callbacks_model_checkpoint_defaults["verbose"]
    )

    parser.add_argument(
        "--callbacks_learning_rate_monitor_log_momentum",
        type=str,
        default=callbacks_learning_rate_monitor_defaults["log_momentum"],
    )
    parser.add_argument(
        "--callbacks_learning_rate_monitor_log_weight_decay",
        type=str,
        default=callbacks_learning_rate_monitor_defaults["log_weight_decay"],
    )

    # Trainer
    parser.add_argument("--max_epochs", type=str, default=trainer_defaults["max_epochs"])
    parser.add_argument("--accelerator", type=str, default=trainer_defaults["accelerator"])
    parser.add_argument("--precision", type=str, default=trainer_defaults["precision"])
    parser.add_argument("--accumulate_grad_batches", type=str, default=trainer_defaults["accumulate_grad_batches"])
    parser.add_argument("--fast_dev_run", type=str, default=trainer_defaults["fast_dev_run"])
    parser.add_argument("--overfit_batches", type=str, default=trainer_defaults["overfit_batches"])

    # Other
    parser.add_argument("--seed", type=str, default=defaults["seed"])
    parser.add_argument("--on_azure", action="store_true")

    return parser


def get_config() -> ClassificationConfig:
    """
    Retrieves configuration for the ClassificationConfig class.

    This function can be extended to load configurations from different sources.

    Returns:
        ClassificationConfig: An instance of SampleConfig with values populated from the command line.
    """
    parser = create_arg_parser()
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    artifacts_dir = Path(args.artifacts_dir) / timestamp

    config_dict: Dict[str, Dict[str, Any]] = {
        "data": {
            "dataset_dir": args.dataset_dir,
            "val_split": str_to_float(args.val_split),
            "test_split": str_to_float(args.test_split),
            "batch_size": str_to_int(args.batch_size),
        },
        "preprocessing": {
            "name_list": [
                getattr(args, f"preprocessing_transform_{i+1}")
                for i in range(5)
                if getattr(args, f"preprocessing_transform_{i+1}") is not None
            ],
            "extra_arguments_list": [
                str_to_dict(getattr(args, f"preprocessing_transform_{i+1}_kwargs"))
                for i in range(5)
                if getattr(args, f"preprocessing_transform_{i+1}") is not None
            ],
        },
        "model": {
            "name": args.model_name,
            "pretrained": str_to_bool(args.pretrained),
            "num_classes": str_to_int(args.num_classes),
            "in_channels": str_to_int(args.in_channels),
        },
        "loss_function": {
            "name": args.loss_function_name,
            "extra_arguments": str_to_dict(args.loss_function_kwargs),
        },
        "optimizer": {
            "name": args.optimizer_name,
            "learning_rate": str_to_float(args.learning_rate),
            "weight_decay": str_to_float(args.weight_decay),
            "extra_arguments": str_to_dict(args.optimizer_kwargs),
        },
        "scheduler": {
            "name": args.scheduler_name,
            "extra_arguments": str_to_dict(args.scheduler_kwargs),
        },
        "metrics": {
            "name_list": [
                getattr(args, f"metric_name_{i+1}") for i in range(5) if getattr(args, f"metric_name_{i+1}") is not None
            ],
            "extra_arguments_list": [
                str_to_dict(getattr(args, f"metric_name_{i+1}_kwargs"))
                for i in range(5)
                if getattr(args, f"metric_name_{i+1}") is not None
            ],
            "task": args.task_type,
            "num_classes": args.num_classes,
        },
        "augmentations": {
            "name_list": [
                getattr(args, f"augmentation_name_{i+1}")
                for i in range(5)
                if getattr(args, f"augmentation_name_{i+1}") is not None
            ],
            "extra_arguments_list": [
                str_to_dict(getattr(args, f"augmentation_name_{i+1}_kwargs"))
                for i in range(5)
                if getattr(args, f"augmentation_name_{i+1}") is not None
            ],
        },
        "callbacks": {
            "early_stopping": {
                "monitor": args.callbacks_early_stopping_monitor,
                "mode": args.callbacks_early_stopping_mode,
                "patience": str_to_int(args.callbacks_early_stopping_patience),
                "min_delta": str_to_float(args.callbacks_early_stopping_min_delta),
                "verbose": str_to_bool(args.callbacks_early_stopping_verbose),
            },
            "model_checkpoint": {
                "monitor": args.callbacks_model_checkpoint_monitor,
                "mode": args.callbacks_model_checkpoint_mode,
                "save_top_k": str_to_int(args.callbacks_model_checkpoint_save_top_k),
                "dirpath": artifacts_dir / "checkpoints",
                "filename": args.callbacks_model_checkpoint_filename,
                "verbose": str_to_bool(args.callbacks_model_checkpoint_verbose),
            },
            "learning_rate_monitor": {
                "log_momentum": str_to_bool(args.callbacks_learning_rate_monitor_log_momentum),
                "log_weight_decay": str_to_bool(args.callbacks_learning_rate_monitor_log_weight_decay),
            },
        },
        "trainer": {
            "max_epochs": str_to_int(args.max_epochs),
            "precision": args.precision,
            "accumulate_grad_batches": str_to_int(args.accumulate_grad_batches),
            "accelerator": args.accelerator,
            "fast_dev_run": str_to_bool(args.fast_dev_run),
            "overfit_batches": str_to_int(args.overfit_batches),
        },
    }

    if args.on_azure:
        loggers_config_dict = {
            "name_list": ["csv", "mlflow"],
            "save_dir": artifacts_dir / "logs",
            "extra_arguments_list": [{}, {}],
        }
    else:
        loggers_config_dict = {
            "name_list": ["csv", "wandb"],
            "save_dir": artifacts_dir / "logs",
            "extra_arguments_list": [
                {},
                {"name": timestamp, "project": PROJECT_NAME},
            ],
        }

    data_config = ClassificationDataConfig(**config_dict["data"])
    preprocessing_config = PreprocessingConfig(**config_dict["preprocessing"])
    model_config = ClassificationModelConfig(**config_dict["model"])
    loss_function_config = ClassificationLossFunctionConfig(**config_dict["loss_function"])
    optimizer_config = OptimizerConfig(**config_dict["optimizer"])
    scheduler_config = (
        SchedulerConfig(**config_dict["scheduler"]) if config_dict["scheduler"]["name"] is not None else None
    )
    metrics_config = ClassificationMetricsConfig(**config_dict["metrics"])
    augmentations_config = AugmentationsConfig(**config_dict["augmentations"])
    callbacks_config = CallbacksConfig(**config_dict["callbacks"])
    loggers_config = LoggersConfig(**loggers_config_dict)
    trainer_config = TrainerConfig(**config_dict["trainer"])

    config = ClassificationConfig(
        data=data_config,
        preprocessing=preprocessing_config,
        model=model_config,
        loss_function=loss_function_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        metrics=metrics_config,
        augmentations=augmentations_config,
        callbacks=callbacks_config,
        loggers=loggers_config,
        trainer=trainer_config,
        seed=str_to_int(args.seed),
    )
    config.log_self()
    config.to_yaml(artifacts_dir / f"config{YAML}")

    return config
