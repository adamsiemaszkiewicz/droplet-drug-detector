# -*- coding: utf-8 -*-
from typing import Any, Dict

from pydantic import BaseModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LRScheduler, StepLR

AVAILABLE_SCHEDULERS: Dict[str, Any] = {
    "cosine_annealing": CosineAnnealingLR,
    "exponential": ExponentialLR,
    "step_lr": StepLR,
}


class SchedulerConfig(BaseModel):
    """Configuration schema for learning rate schedulers.

    Attributes:
        name: A string indicating the name of the scheduler to be used.
        arguments: A dictionary containing all scheduler-specific arguments.
    """

    name: str
    arguments: Dict[str, Any] = {}


def create_scheduler(config: SchedulerConfig, optimizer: Optimizer) -> LRScheduler:
    """Create a learning rate scheduler based on the configuration.

    Args:
        config: Configuration object specifying the type and parameters of the scheduler.
        optimizer: The optimizer for which to schedule the learning rate.

    Returns:
        An instance of a PyTorch learning rate scheduler.

    Raises:
        ValueError: If an invalid scheduler name is provided.
    """
    if config.name not in AVAILABLE_SCHEDULERS:
        raise NotImplementedError(f"Scheduler {config.name} is not recognized or supported.")

    scheduler_class = AVAILABLE_SCHEDULERS[config.name]
    scheduler = scheduler_class(optimizer, **config.arguments)

    return scheduler
