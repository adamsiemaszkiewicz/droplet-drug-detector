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

REQUIRED_ARGUMENTS: Dict[str, str] = {
    "cosine_annealing": "T_max",
    "exponential": "gamma",
    "one_cycle": "max_lr",
    "step_lr": "step_size",
}


class SchedulerConfig(BaseModel):
    """
    Configuration for creating a learning rate scheduler.

    Attributes:
        name (str): A string indicating the name of the scheduler to be used.
        extra_arguments (Optional[Dict[str, Any]): A dictionary containing all other scheduler-specific arguments.
    """

    name: str
    extra_arguments: Dict[str, Any] = {}

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validates if the optimizer is implemented.
        """
        if v not in AVAILABLE_SCHEDULERS:
            raise ValueError(
                f"Scheduler '{v}' is not implemented.\nAvailable schedulers: {list(AVAILABLE_SCHEDULERS.keys())}"
            )
        return v

    @validator("extra_arguments", pre=True, always=True)
    def validate_required_arguments(cls, v: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates if the required arguments for the chosen scheduler are provided.
        """
        scheduler_name = values.get("name")
        if scheduler_name in REQUIRED_ARGUMENTS:
            required_arg = REQUIRED_ARGUMENTS[scheduler_name]
            if required_arg not in v:
                raise ValueError(f"Required argument '{required_arg}' for scheduler '{scheduler_name}' is missing.")
        return v


def create_scheduler(config: SchedulerConfig, optimizer: Optimizer) -> LRScheduler:
    """
    Create a learning rate scheduler based on the configuration.

    Args:
        config (SchedulerConfig): A SchedulerConfig instance containing the scheduler name and extra arguments.
        optimizer (Optimizer): An Optimizer instance for which the scheduler will manage the learning rate.

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
    scheduler = scheduler_class(optimizer, **config.extra_arguments)

    _logger.info("Learning rate scheduler successfully created.")

    return scheduler
