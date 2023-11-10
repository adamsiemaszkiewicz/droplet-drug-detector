# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from typing import Any, Dict

from src.sample.aml.components.sample_component.config import SampleConfig


def create_arg_parser() -> ArgumentParser:
    """
    Creates an argument parser for SampleConfig class.

    Returns:
        ArgumentParser: The argument parser with configuration options.
    """
    parser = ArgumentParser(description="Parse configuration for the SampleConfig class")

    parser.add_argument("--parameter_1", type=str, default="parameter-1", help="Sample parameter 1")
    parser.add_argument("--parameter_2", type=str, default="42", help="Sample parameter 2")
    parser.add_argument("--parameter_3", type=str, default="0.52", help="Sample parameter 3")
    parser.add_argument("--parameter_4", type=str, default=".", help="Sample parameter 4")
    parser.add_argument("--parameter_5", type=str, default="parameter,_,5 ", help="Sample parameter 5")
    parser.add_argument("--parameter_6", type=str, default="1, 2,3", help="Sample parameter 6")
    parser.add_argument("--parameter_7", type=str, default=" 0.1,0.2,3e-4", help="Sample parameter 7")
    parser.add_argument("--parameter_8", type=str, default="True", help="Sample parameter 8")
    parser.add_argument("--parameter_9", action="store_true", help="Sample parameter 7")

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

    config = SampleConfig(**config_dict)
    config.log_self()

    return config
