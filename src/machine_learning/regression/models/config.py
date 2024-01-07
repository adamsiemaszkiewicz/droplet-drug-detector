# -*- coding: utf-8 -*-
from pydantic import BaseModel, validator

from src.common.utils.logger import get_logger
from src.machine_learning.classification.models.types import AVAILABLE_MODELS

_logger = get_logger(__name__)


class RegressionModelConfig(BaseModel):
    """
    Configuration for creating a regression model.

    Attrs:
        name: The name of the model architecture.
        pretrained: Whether to use pretrained weights.
        num_classes: Number of classes in the dataset.
        in_channels: Number of input channels to the model.
    """

    name: str
    pretrained: bool
    num_classes: int
    in_channels: int

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validates if the model is implemented.
        """
        if v not in AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{v}' is not implemented.\nAvailable model architecture: https://huggingface.co/timm"
            )
        return v

    @validator("num_classes", "in_channels")
    def validate_positive_integer(cls, v: int) -> int:
        """
        Validates if the provided value is a positive integer.
        """
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"The value {v} must be a positive integer.")
        return v
