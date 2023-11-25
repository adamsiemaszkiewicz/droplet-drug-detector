# -*- coding: utf-8 -*-
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.common.utils.logger import get_logger
from src.machine_learning.scheduler.config import SchedulerConfig
from src.machine_learning.scheduler.types import AVAILABLE_SCHEDULERS

_logger = get_logger(__name__)


def create_scheduler(config: SchedulerConfig, optimizer: Optimizer, total_steps: int) -> LRScheduler:
    """
    Create a learning rate scheduler based on the configuration.

    Args:
        config (SchedulerConfig): A SchedulerConfig instance containing the scheduler name and extra arguments.
        optimizer (Optimizer): An Optimizer instance for which the scheduler will manage the learning rate.
        total_steps (int): Total number of training steps required for some schedulers like OneCycleLR.

    Returns:
        LRScheduler: A PyTorch learning rate scheduler.
    """
    _logger.info(f"Creating learning rate scheduler with the following configuration: {config.dict()}")

    scheduler_class = AVAILABLE_SCHEDULERS.get(config.name)
    if scheduler_class is None:
        raise ValueError(
            f"Scheduler '{config.name}' is not implemented.\n"
            f"Available schedulers: {list(AVAILABLE_SCHEDULERS.keys())}"
        )

    if config.name == "one_cycle":
        config.extra_arguments["total_steps"] = total_steps

    scheduler = scheduler_class(optimizer, **config.extra_arguments)

    _logger.info("Learning rate scheduler successfully created.")

    return scheduler
