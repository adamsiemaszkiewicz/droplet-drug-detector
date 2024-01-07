# -*- coding: utf-8 -*-
from typing import Any, Dict

from pydantic import BaseModel, validator

from src.machine_learning.regression.loss_functions.types import AVAILABLE_LOSS_FUNCTIONS


class RegressionLossFunctionConfig(BaseModel):
    """
    Configuration for creating a loss function.

    Attrs:
        name: The name of the loss function.
        extra_arguments: A dictionary containing all other loss function-specific arguments.
    """

    name: str
    extra_arguments: Dict[str, Any] = {}

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validates if the loss function is implemented.
        """
        if v not in AVAILABLE_LOSS_FUNCTIONS:
            raise ValueError(
                f"Loss function '{v}' is not implemented.\n"
                f"Available loss functions: {list(AVAILABLE_LOSS_FUNCTIONS.keys())}"
            )
        return v
