# -*- coding: utf-8 -*-
from typing import Any, Dict, Iterator, Optional, Type

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
    extra_arguments: Optional[Dict[str, Any]] = None

    @validator("name")
    def validate_names(cls, v: str) -> str:
        """
        Validator to ensure each name in `name_list` corresponds to a valid augmentation.
        """
        if v not in AVAILABLE_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{v}' is not implemented. Available optimizers: {list(AVAILABLE_OPTIMIZERS.keys())}"
            )
        return v

    @validator("extra_arguments", pre=True, always=True)
    def check_default_extra_args(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates the 'extra_arguments' by assigning a default empty dictionary if None is provided,
        and also checks necessary keys for different scheduler types.
        """
        if v is None:
            v = {}
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
