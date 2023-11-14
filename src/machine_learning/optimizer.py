# -*- coding: utf-8 -*-
from typing import Any, Dict, Iterator, Type

from pydantic import BaseModel, validator
from torch.nn.parameter import Parameter
from torch.optim import SGD, Adam, Optimizer

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_OPTIMIZERS: Dict[str, Type[Optimizer]] = {
    "adam": Adam,
    "sgd": SGD,
}


class OptimizerConfig(BaseModel):
    """
    Configuration schema for optimizers with explicit common fields and provisions for additional arguments.
    """

    name: str
    learning_rate: float
    weight_decay: float = 0
    extra_arguments: Dict[str, Any] = {}

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validates if the optimizer is implemented.
        """
        if v not in AVAILABLE_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{v}' is not implemented.\n" f"Available optimizers: {list(AVAILABLE_OPTIMIZERS.keys())}"
            )
        return v

    @validator("learning_rate")
    def validate_learning_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Learning rate must be a positive number.")
        return v

    @validator("weight_decay")
    def validate_weight_decay(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Weight decay must be a non-negative number.")
        return v


def create_optimizer(config: OptimizerConfig, parameters: Iterator[Parameter]) -> Optimizer:
    """
    Create an optimizer based on the configuration provided in OptimizerConfig.
    This function unpacks common arguments explicitly and passes any additional arguments
    found in the `extra_arguments` attribute of the configuration.

    Args:
        config: Configuration object specifying the type and parameters of the optimizer.
        parameters: An iterator over the model parameters (typically the result of model.parameters()).

    Returns:
        An instance of a PyTorch Optimizer.
    """
    if config.extra_arguments is None:
        raise ValueError("'extra_arguments' cannot be None")

    _logger.info(
        f"Creating optimizer sequence with the following configuration: "
        f"weight_decay={config.weight_decay}, "
        f"learning_rate={config.learning_rate}, "
        f"extra_arguments={config.extra_arguments}"
    )

    optimizer_class = AVAILABLE_OPTIMIZERS[config.name]
    optimizer_args = {"lr": config.learning_rate, "weight_decay": config.weight_decay, **config.extra_arguments}
    optimizer = optimizer_class(params=parameters, **optimizer_args)

    return optimizer
