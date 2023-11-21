# -*- coding: utf-8 -*-
from typing import Iterator

from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from src.common.utils.logger import get_logger
from src.machine_learning.optimizer.config import OptimizerConfig
from src.machine_learning.optimizer.types import AVAILABLE_OPTIMIZERS

_logger = get_logger(__name__)


def create_optimizer(config: OptimizerConfig, parameters: Iterator[Parameter]) -> Optimizer:
    """
    Create an optimizer based on the configuration.

    Args:
        config (OptimizerConfig): Configuration object containing optimizer parameters.
        parameters (Iterator[Parameter]: Mmodel parameters iterator (typically the result of model.parameters()).

    Returns:
        Optimizer: A PyTorch optimizer.
    """
    _logger.info(f"Creating optimizer with the following configuration: {config.dict()}")

    optimizer_class = AVAILABLE_OPTIMIZERS[config.name]
    optimizer_arguments = {"lr": config.learning_rate, "weight_decay": config.weight_decay, **config.extra_arguments}
    optimizer = optimizer_class(params=parameters, **optimizer_arguments)

    _logger.info("Optimizer successfully created.")

    return optimizer
