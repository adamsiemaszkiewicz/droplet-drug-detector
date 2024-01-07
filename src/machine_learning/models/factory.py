# -*- coding: utf-8 -*-
import timm
from torch.nn import Module

from src.common.utils.logger import get_logger
from src.machine_learning.models.config import BaseModelConfig

_logger = get_logger(__name__)


def create_model(config: BaseModelConfig) -> Module:
    """
    Create a deep learning model based on the configuration.

    Args:
        config (BaseModelConfig): Configuration object containing model parameters.

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
