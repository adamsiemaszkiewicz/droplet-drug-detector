# -*- coding: utf-8 -*-
from torch.nn import Module

from src.common.utils.logger import get_logger
from src.machine_learning.classification.loss_functions.config import ClassificationLossFunctionConfig
from src.machine_learning.classification.loss_functions.types import AVAILABLE_LOSS_FUNCTIONS

_logger = get_logger(__name__)


def create_loss_function(config: ClassificationLossFunctionConfig) -> Module:
    """
    Create a loss function based on the configuration.

    Args:
        config (ClassificationLossFunctionConfig): Configuration object containing loss function parameters.

    Returns:
        Module: A PyTorch loss function.
    """
    _logger.info(f"Creating loss function with the following configuration: {config.dict()}")

    loss_class = AVAILABLE_LOSS_FUNCTIONS.get(config.name)
    if loss_class is None:
        raise ValueError(
            f"Loss function '{config.name}' is not implemented.\n"
            f"Available loss functions: {list(AVAILABLE_LOSS_FUNCTIONS.keys())}"
        )

    if "reduction" in loss_class.__init__.__code__.co_varnames:
        config.extra_arguments["reduction"] = "none"  # Set 'reduction' to 'none' if supported
    else:
        _logger.warning(f"Loss function '{config.name}' does not support 'reduction' argument.")

    loss_function = loss_class(**config.extra_arguments)

    _logger.info("Loss function successfully created.")

    return loss_function
