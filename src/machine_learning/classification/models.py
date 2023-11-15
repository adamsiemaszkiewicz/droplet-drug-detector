# -*- coding: utf-8 -*-
from typing import List

import timm
from pydantic import validator
from torch.nn import Module

from src.common.utils.logger import get_logger
from src.configs.base import BaseModelConfig

_logger = get_logger(__name__)

AVAILABLE_MODELS: List[str] = timm.list_models()


class ClassificationModelConfig(BaseModelConfig):
    """
    Configuration for creating a classification model.

    Attributes:
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


def create_model(config: ClassificationModelConfig) -> Module:
    """
    Create a classification model based on the configuration.
    List of available architectures: https://huggingface.co/timm

    Args:
        config (ClassificationModelConfig): Configuration object containing model parameters.

    Returns:
        Module: A PyTorch model.
    """
    _logger.info(f"Creating model with the following configuration: {config.dict()}")

    model = timm.create_model(
        model_name=config.name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        in_chans=config.in_channels,
    )

    _logger.info("Model successfully created.")

    return model
