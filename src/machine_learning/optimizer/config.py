# -*- coding: utf-8 -*-
from typing import Any, Dict

from pydantic import validator

from src.common.utils.logger import get_logger
from src.configs.base import BaseOptimizerConfig
from src.machine_learning.optimizer.types import AVAILABLE_OPTIMIZERS

_logger = get_logger(__name__)


class OptimizerConfig(BaseOptimizerConfig):
    """
    Configuration for creating optimizer.

    Attributes:
        name: The name of the optimizer.
        learning_rate: The learning rate of the optimizer.
        weight_decay: The weight decay of the optimizer.
        extra_arguments: A dictionary containing all other loss optimizer-specific arguments.
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
                f"Optimizer '{v}' is not implemented.\nAvailable optimizers: {list(AVAILABLE_OPTIMIZERS.keys())}"
            )
        return v

    @validator("learning_rate")
    def validate_learning_rate(cls, v: float) -> float:
        """
        Validates if the learning rate is a positive number.
        """
        if v <= 0:
            raise ValueError("Learning rate must be a positive number.")
        return v

    @validator("weight_decay")
    def validate_weight_decay(cls, v: float) -> float:
        """
        Validates if the weight decay is a non-negative number.
        """
        if v < 0:
            raise ValueError("Weight decay must be a non-negative number.")
        return v
