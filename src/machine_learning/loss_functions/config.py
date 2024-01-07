# -*- coding: utf-8 -*-
from typing import Any, Dict

from pydantic import BaseModel, validator

from src.machine_learning.loss_functions.types import CLASSIFICATION_LOSS_FUNCTIONS, REGRESSION_LOSS_FUNCTIONS


class BaseLossFunctionConfig(BaseModel):
    """
    Configuration for creating a loss function.

    Attrs:
        name: The name of the loss function.
        extra_arguments: A dictionary containing all other loss function-specific arguments.
    """

    name: str
    extra_arguments: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class ClassificationLossFunctionConfig(BaseLossFunctionConfig):
    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validates if the loss function is implemented.
        """
        if v not in CLASSIFICATION_LOSS_FUNCTIONS:
            raise ValueError(
                f"Loss function '{v}' is not implemented.\n"
                f"Available loss functions: {list(CLASSIFICATION_LOSS_FUNCTIONS.keys())}"
            )
        return v


class RegressionLossFunctionConfig(BaseLossFunctionConfig):
    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validates if the loss function is implemented.
        """
        if v not in REGRESSION_LOSS_FUNCTIONS:
            raise ValueError(
                f"Loss function '{v}' is not implemented.\n"
                f"Available loss functions: {list(REGRESSION_LOSS_FUNCTIONS.keys())}"
            )
        return v
