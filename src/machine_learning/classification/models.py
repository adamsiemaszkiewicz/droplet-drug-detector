# -*- coding: utf-8 -*-
from typing import Any, Dict, List

import timm
from pydantic import BaseModel, conint, validator
from torch.nn import Module

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_MODELS: List[str] = timm.list_models()
AVAILABLE_PRETRAINED_MODELS: List[str] = timm.list_models(pretrained=True)


class ClassificationModelConfig(BaseModel):
    """
    Configuration for creating a classification model.
    """

    name: str
    pretrained: bool
    num_classes: int = conint(gt=0)  # Ensure a positive integer
    in_channels: int = conint(gt=0)  # Ensure a positive integer

    @validator("name")
    def validate_model_name(cls, v: str) -> str:
        """
        Validates if the model name is available.
        """
        if v not in AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{v}' is not implemented. Check available architectures at https://huggingface.co/timm"
            )
        return v

    @validator("pretrained")
    def validate_pretrained(cls, v: bool, values: Dict[str, Any]) -> bool:
        """
        Validates that the model supports pretrained weights if pretrained is set to True.
        """
        if v and "name" in values and values["name"] not in AVAILABLE_PRETRAINED_MODELS:
            raise ValueError(f"Pretrained weights not available for model '{values['name']}'.")
        return v


def create_model(config: ClassificationModelConfig) -> Module:
    """
    Create a classification model based on a configuration instance.
    List of available architectures: https://huggingface.co/timm

    Args:
        config (ClassificationModelConfig): Configuration object containing model parameters.

    Returns:
        Module: The created model.
    """
    model = timm.create_model(
        model_name=config.name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        in_chans=config.in_channels,
    )

    return model
