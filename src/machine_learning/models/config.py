# -*- coding: utf-8 -*-
from pydantic import BaseModel, validator

from src.common.utils.logger import get_logger
from src.machine_learning.models.types import CLASSIFICATION_MODELS, REGRESSION_MODELS

_logger = get_logger(__name__)


class BaseModelConfig(BaseModel):
    """
    Configuration for creating a deep learning model.

    Attrs:
        name: The name of the model architecture.
        pretrained: Whether to use pretrained weights.
        in_channels: Number of input channels to the model.
        num_classes: Number of classes in the dataset.
    """

    name: str
    pretrained: bool
    in_channels: int
    num_classes: int

    @validator("num_classes", "in_channels")
    def validate_positive_integer(cls, v: int) -> int:
        """
        Validates if the provided value is a positive integer.
        """
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"The value {v} must be a positive integer.")
        return v


class ClassificationModelConfig(BaseModelConfig):
    """
    Configuration for creating a classification model.
    """

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validates if the model is implemented.
        """
        if v not in CLASSIFICATION_MODELS:
            raise ValueError(
                f"Model '{v}' is not implemented.\nAvailable model architecture: https://huggingface.co/timm"
            )
        return v


class RegressionModelConfig(BaseModelConfig):
    """
    Configuration for creating a regression model.
    """

    num_classes: int = 1

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validates if the model is implemented.
        """
        if v not in REGRESSION_MODELS:
            raise ValueError(
                f"Model '{v}' is not implemented.\nAvailable model architecture: https://huggingface.co/timm"
            )
        return v
