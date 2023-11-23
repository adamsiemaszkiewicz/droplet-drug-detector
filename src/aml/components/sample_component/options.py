# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from typing import Any, Dict

from src.aml.components.sample_component.config import (
    SampleCallbacksConfig,
    SampleConfig,
    SampleDataConfig,
    SampleModelConfig,
    SampleTrainerConfig,
)


def create_arg_parser() -> ArgumentParser:
    """
    Creates an argument parser for SampleConfig class.

    Returns:
        ArgumentParser: The argument parser with configuration options.
    """
    parser = ArgumentParser(description="Parse configuration for the SampleConfig class")

    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--val_split", type=str)
    parser.add_argument("--test_split", type=str)
    parser.add_argument("--batch_size", type=str)
    parser.add_argument("--preprocessing_transform_1", type=str)
    parser.add_argument("--preprocessing_transform_1_kwargs", type=str)
    parser.add_argument("--preprocessing_transform_2", type=str)
    parser.add_argument("--preprocessing_transform_2_kwargs", type=str)
    parser.add_argument("--preprocessing_transform_3", type=str)
    parser.add_argument("--preprocessing_transform_3_kwargs", type=str)
    parser.add_argument("--preprocessing_transform_4", type=str)
    parser.add_argument("--preprocessing_transform_4_kwargs", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--pretrained", type=bool)
    parser.add_argument("--num_classes", type=str)
    parser.add_argument("--in_channels", type=str)
    parser.add_argument("--loss_function_name", type=str)
    parser.add_argument("--loss_function_kwargs", type=str)
    parser.add_argument("--optimizer_name", type=str)
    parser.add_argument("--learning_rate", type=str)
    parser.add_argument("--weight_decay", type=str)
    parser.add_argument("--optimizer_kwargs", type=str)
    parser.add_argument("--scheduler_name", type=str)
    parser.add_argument("--scheduler_kwargs", type=str)
    parser.add_argument("--task_type", type=str)
    parser.add_argument("--metric_name_1", type=str)
    parser.add_argument("--metric_name_1_kwargs", type=str)
    parser.add_argument("--metric_name_2", type=str)
    parser.add_argument("--metric_name_2_kwargs", type=str)
    parser.add_argument("--metric_name_3", type=str)
    parser.add_argument("--metric_name_3_kwargs", type=str)
    parser.add_argument("--metric_name_4", type=str)
    parser.add_argument("--metric_name_4_kwargs", type=str)
    parser.add_argument("--metric_name_5", type=str)
    parser.add_argument("--metric_name_5_kwargs", type=str)
    parser.add_argument("--augmentation_name_1", type=str)
    parser.add_argument("--augmentation_name_1_kwargs", type=str)
    parser.add_argument("--augmentation_name_2", type=str)
    parser.add_argument("--augmentation_name_2_kwargs", type=str)
    parser.add_argument("--augmentation_name_3", type=str)
    parser.add_argument("--augmentation_name_3_kwargs", type=str)
    parser.add_argument("--augmentation_name_4", type=str)
    parser.add_argument("--augmentation_name_4_kwargs", type=str)
    parser.add_argument("--augmentation_name_5", type=str)
    parser.add_argument("--augmentation_name_5_kwargs", type=str)
    parser.add_argument("--max_epochs", type=str)
    parser.add_argument("--patience", type=str)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--accumulate_grad_batches", type=str)
    parser.add_argument("--fast_dev_run", type=bool)
    parser.add_argument("--overfit_batches", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--on_azure", action="store_true")

    return parser


def get_config() -> SampleConfig:
    """
    Retrieves configuration for the SampleConfig class.

    This function can be extended to load configurations from different sources.

    Returns:
        SampleConfig: An instance of SampleConfig with values populated from the command line.
    """
    parser = create_arg_parser()
    args = parser.parse_args()

    config_dict: Dict[str, Dict[str, Any]] = {
        "data": {
            "parameter_1": args.parameter_1,
            "parameter_2": args.parameter_2,
            "parameter_3": args.parameter_3,
            "parameter_4": args.parameter_4,
            "parameter_5": args.parameter_5,
            "parameter_6": args.parameter_6,
            "parameter_7": args.parameter_7,
            "parameter_8": args.parameter_8,
        },
        "model": {
            "parameter_1": args.parameter_1,
            "parameter_2": args.parameter_2,
            "parameter_3": args.parameter_3,
            "parameter_4": args.parameter_4,
            "parameter_5": args.parameter_5,
            "parameter_6": args.parameter_6,
            "parameter_7": args.parameter_7,
            "parameter_8": args.parameter_8,
        },
        "trainer": {
            "parameter_1": args.parameter_1,
            "parameter_2": args.parameter_2,
            "parameter_3": args.parameter_3,
            "parameter_4": args.parameter_4,
            "parameter_5": args.parameter_5,
            "parameter_6": args.parameter_6,
            "parameter_7": args.parameter_7,
            "parameter_8": args.parameter_8,
        },
        "callbacks": {
            "parameter_1": args.parameter_1,
            "parameter_2": args.parameter_2,
            "parameter_3": args.parameter_3,
            "parameter_4": args.parameter_4,
            "parameter_5": args.parameter_5,
            "parameter_6": args.parameter_6,
            "parameter_7": args.parameter_7,
            "parameter_8": args.parameter_8,
        },
    }

    data_config = SampleDataConfig(**config_dict["data"])
    model_config = SampleModelConfig(**config_dict["model"])
    trainer_config = SampleTrainerConfig(**config_dict["trainer"])
    callbacks_config = SampleCallbacksConfig(**config_dict["callbacks"])

    config = SampleConfig(data=data_config, model=model_config, trainer=trainer_config, callbacks=callbacks_config)
    config.log_self()

    return config
