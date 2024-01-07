# -*- coding: utf-8 -*-
from torch.nn import Module

from src.common.utils.logger import get_logger
from src.machine_learning.loss_functions.config import (
    BaseLossFunctionConfig,
    ClassificationLossFunctionConfig,
    RegressionLossFunctionConfig,
)
from src.machine_learning.loss_functions.types import CLASSIFICATION_LOSS_FUNCTIONS, REGRESSION_LOSS_FUNCTIONS

_logger = get_logger(__name__)


def create_loss_function(config: BaseLossFunctionConfig) -> Module:
    """
    Create a loss function based on the configuration.

    Args:
        config (ClassificationLossFunctionConfig): Configuration object containing loss function parameters.

    Returns:
        Module: A PyTorch loss function.
    """
    _logger.info(f"Creating loss function with the following configuration: {config.dict()}")

    if isinstance(config, ClassificationLossFunctionConfig):
        loss_class = CLASSIFICATION_LOSS_FUNCTIONS.get(config.name)
    elif isinstance(config, RegressionLossFunctionConfig):
        loss_class = REGRESSION_LOSS_FUNCTIONS.get(config.name)
    else:
        raise ValueError(f"Unsupported loss function config type: {type(config)}")

    if loss_class is None:
        raise ValueError(f"Loss function '{config.name}' is not implemented for {type(config).__name__}.\n")

    if "reduction" in loss_class.__init__.__code__.co_varnames:
        config.extra_arguments["reduction"] = "none"  # Set 'reduction' to 'none' if supported
    else:
        _logger.warning(f"Loss function '{config.name}' does not support 'reduction' argument.")

    loss_function = loss_class(**config.extra_arguments)

    _logger.info("Loss function successfully created.")

    return loss_function
