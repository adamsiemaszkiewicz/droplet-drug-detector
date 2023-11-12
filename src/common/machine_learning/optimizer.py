# -*- coding: utf-8 -*-
from typing import Any, Dict, Type

from pydantic import BaseModel
from torch.optim import SGD, Adam, Optimizer

AVAILABLE_OPTIMIZERS: Dict[str, Type[Optimizer]] = {
    "adam": Adam,
    "sgd": SGD,
}


class OptimizerConfig(BaseModel):
    """
    Configuration schema for optimizers with explicit common fields and provisions for additional arguments.

    Attributes:
        name: The name of the optimizer.
        learning_rate: The learning rate.
        extra_arguments: A dictionary containing all other optimizer-specific arguments.
    """

    name: str
    learning_rate: float
    weight_decay: float = 0
    extra_arguments: Dict[str, Any]


def create_optimizer(config: OptimizerConfig) -> Optimizer:
    """
    Create an optimizer based on the configuration provided in OptimizerConfig.
    This function unpacks common arguments explicitly and passes any additional arguments
    found in the `extra_arguments` attribute of the configuration.

    Args:
        config: Configuration object specifying the type and parameters of the optimizer.

    Returns:
        An instance of a PyTorch Optimizer.

    Raises:
        ValueError: If an invalid optimizer type is provided or required parameters are missing.
    """
    if config.name not in AVAILABLE_OPTIMIZERS:
        raise ValueError(
            f"Optimizer '{config.name}' is not implemented. Available optimizers: {list(AVAILABLE_OPTIMIZERS.keys())}"
        )

    optimizer_class = AVAILABLE_OPTIMIZERS[config.name]
    optimizer_args = {"lr": config.learning_rate, "weight_decay": config.weight_decay, **config.extra_arguments}
    optimizer = optimizer_class(**optimizer_args)

    return optimizer
