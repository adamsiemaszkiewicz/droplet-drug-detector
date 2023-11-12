# -*- coding: utf-8 -*-
from typing import Any, Dict, Type

from pydantic import BaseModel, validator
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
        extra_arguments: A dictionary containing all other scheduler-specific arguments that are not commonly used.
    """

    name: str
    extra_arguments: Dict[str, Any]

    @validator("extra_arguments")
    def check_default_extra_args(cls, v: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the 'extra_arguments' if necessary keys are present for different types of schedulers.
        """
        if "name" in values and values["name"] == "cosine_annealing" and "T_max" not in v:
            raise ValueError("T_max must be provided for CosineAnnealingLR")
        if "name" in values and values["name"] == "exponential" and "gamma" not in v:
            raise ValueError("gamma must be provided for ExponentialLR")
        if "name" in values and values["name"] == "one_cycle" and "max_lr" not in v:
            raise ValueError("max_lr must be provided for OneCycleLR")
        if "name" in values and values["name"] == "step_lr" and "step_size" not in v:
            raise ValueError("step_size must be provided for StepLR")
        return v


def create_scheduler(config: SchedulerConfig, optimizer: Optimizer) -> LRScheduler:
    """
    Factory function to create a learning rate scheduler based on provided configuration.

    Args:
        config: A SchedulerConfig instance containing the scheduler name and extra arguments.
        optimizer: An Optimizer instance for which the scheduler will manage the learning rate.

    Returns:
        An instantiated learning rate scheduler object.

    Raises:
        ValueError: If the scheduler name is not recognized, or if required arguments are missing or incorrect.
    """
    scheduler_class = AVAILABLE_SCHEDULERS.get(config.name)
    if scheduler_class is None:
        valid_schedulers = list(AVAILABLE_SCHEDULERS.keys())
        error_message = f"Scheduler '{config.name}' is not implemented. Available schedulers: {valid_schedulers}"
        _logger.error(error_message)
        raise ValueError(error_message)

    try:
        scheduler = scheduler_class(optimizer, **config.extra_arguments)
        _logger.info(f"Scheduler '{config.name}' created with arguments: {config.extra_arguments}")
    except TypeError as e:
        error_message = f"Incorrect arguments for scheduler '{config.name}'. Error: {e}"
        _logger.error(error_message)
        raise ValueError(error_message)

    return scheduler
