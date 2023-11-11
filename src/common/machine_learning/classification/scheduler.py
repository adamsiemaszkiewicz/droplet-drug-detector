# -*- coding: utf-8 -*-
from typing import Any, Dict, Type

from pydantic import BaseModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LRScheduler, OneCycleLR, StepLR

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_SCHEDULERS: Dict[str, Type[LRScheduler]] = {
    "cosine_annealing": CosineAnnealingLR,
    "exponential": ExponentialLR,
    "one_cycle": OneCycleLR,
    "step_lr": StepLR,
}


class SchedulerConfig(BaseModel):
    """
    Configuration for creating a learning rate scheduler.

    Attributes:
        name: A string indicating the name of the scheduler to be used.
        arguments: A dictionary containing all scheduler-specific arguments.
    """

    name: str
    arguments: Dict[str, Any]


def create_scheduler(config: SchedulerConfig, optimizer: Optimizer) -> LRScheduler:
    """
    Create a learning rate scheduler based on the configuration.

    Args:
        config: Configuration object specifying the type and parameters of the scheduler.
        optimizer: The optimizer for which to schedule the learning rate.

    Returns:
        LRScheduler: An instance of a PyTorch learning rate scheduler.

    Raises:
        ValueError: If an invalid scheduler name is provided.
    """
    scheduler_class = AVAILABLE_SCHEDULERS.get(config.name)
    if scheduler_class is None:
        valid_schedulers = list(AVAILABLE_SCHEDULERS.keys())
        error_message = f"Scheduler '{config.name}' is not implemented. Available schedulers: {valid_schedulers}"
        _logger.error(error_message)
        raise ValueError(error_message)

    try:
        scheduler = scheduler_class(optimizer, **config.arguments)
        _logger.info(f"Scheduler '{config.name}' created with arguments: {config.arguments}")
    except TypeError as e:
        error_message = f"Incorrect arguments for scheduler '{config.name}'. Error: {e}"
        _logger.error(error_message)
        raise ValueError(error_message)

    return scheduler
