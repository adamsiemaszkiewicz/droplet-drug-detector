# -*- coding: utf-8 -*-
import timm
from torch.nn import Module

from src.common.utils.logger import get_logger
from src.machine_learning.regression.models.config import RegressionModelConfig

_logger = get_logger(__name__)


def create_model(config: RegressionModelConfig) -> Module:
    """
    Create a regression model based on the configuration adapted for single-value regression tasks.
    List of available architectures: https://huggingface.co/timm

    Args:
        config (RegressionModelConfig): Configuration object containing model parameters.

    Returns:
        Module: A PyTorch model adapted for regression tasks.
    """
    _logger.info(f"Creating regression model with the following configuration: {config.dict()}")

    # Create the base model
    model = timm.create_model(
        model_name=config.name,
        pretrained=config.pretrained,
        num_classes=1,  # Predict a single value
        in_chans=config.in_channels,
    )

    _logger.info("Regression model successfully created.")

    return model
