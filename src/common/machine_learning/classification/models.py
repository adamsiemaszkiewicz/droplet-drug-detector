# -*- coding: utf-8 -*-
import timm
from pydantic import BaseModel
from torch.nn import Module

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class ModelConfig(BaseModel):
    """Configuration schema to instantiate models from timm."""

    model_name: str
    pretrained: bool
    num_classes: int
    in_channels: int


def create_model(config: ModelConfig) -> Module:
    """
    Create a timm model based on a configuration instance.

    List of available architectures: https://huggingface.co/timm

    Args:
        config (ModelConfig): Configuration object containing model parameters.

    Returns:
        Module: The created model.

    Raises:
            NotImplementedError: If the model_name is not recognized.
    """
    _logger.info(
        f"Creating {'pretrained ' if config.pretrained else ''}model {config.model_name} "
        f"with {config.num_classes} classes & {config.in_channels} input channels."
    )

    available_architectures = timm.list_models()

    if config.model_name not in available_architectures:
        raise NotImplementedError(
            f"Model {config.model_name} not found in timm. Check available architectures at https://huggingface.co/timm"
        )

    model = timm.create_model(
        model_name=config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        in_chans=config.in_channels,
    )
    return model
