# -*- coding: utf-8 -*-
from typing import List

import timm
from pydantic import BaseModel
from torch.nn import Module

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_CLASSIFICATION_MODELS: List[str] = timm.list_models()


class ClassificationModelConfig(BaseModel):
    """
    Configuration for creating a timm model.
    """

    name: str
    pretrained: bool
    num_classes: int
    in_channels: int


def create_model(config: ClassificationModelConfig) -> Module:
    """
    Create a timm model based on a configuration instance.

    List of available architectures: https://huggingface.co/timm

    Args:
        config (ClassificationModelConfig): Configuration object containing model parameters.

    Returns:
        Module: The created model.

    Raises:
        NotImplementedError: If the model_name is not recognized.
    """
    if config.name not in AVAILABLE_CLASSIFICATION_MODELS:
        raise NotImplementedError(
            f"Model '{config.name}' not found in timm. Check available architectures at https://huggingface.co/timm"
        )

    model = timm.create_model(
        model_name=config.name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        in_chans=config.in_channels,
    )

    return model
