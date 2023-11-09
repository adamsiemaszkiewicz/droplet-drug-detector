# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from typing import Any, Dict

from src.components.sample_component.config import SampleConfig


def create_arg_parser() -> ArgumentParser:
    """
    Creates an argument parser for SampleConfig class.

    Returns:
        ArgumentParser: The argument parser with configuration options.
    """
    parser = ArgumentParser(description="Parse configuration for the SampleConfig class")

    parser.add_argument("--parameter-1", type=str, help="Sample parameter 1")
    parser.add_argument("--parameter-2", type=str, help="Sample parameter 2")
    parser.add_argument("--parameter-3", type=str, help="Sample parameter 3")
    parser.add_argument("--parameter-4", type=str, help="Sample parameter 4")
    parser.add_argument("--parameter-5", type=str, help="Sample parameter 5")
    parser.add_argument("--parameter-6", action="store_true", help="Sample parameter 6")

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
        },
        "model": {
            "parameter_1": args.parameter_1,
            "parameter_2": args.parameter_2,
            "parameter_3": args.parameter_3,
            "parameter_4": args.parameter_4,
            "parameter_5": args.parameter_5,
            "parameter_6": args.parameter_6,
        },
        "trainer": {
            "parameter_1": args.parameter_1,
            "parameter_2": args.parameter_2,
            "parameter_3": args.parameter_3,
            "parameter_4": args.parameter_4,
            "parameter_5": args.parameter_5,
            "parameter_6": args.parameter_6,
        },
        "callbacks": {
            "parameter_1": args.parameter_1,
            "parameter_2": args.parameter_2,
            "parameter_3": args.parameter_3,
            "parameter_4": args.parameter_4,
            "parameter_5": args.parameter_5,
            "parameter_6": args.parameter_6,
        },
    }

    config = SampleConfig(**config_dict)
    config.log_self()

    return config
