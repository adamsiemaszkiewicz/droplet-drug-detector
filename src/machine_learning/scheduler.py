# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Type

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
        name (str): A string indicating the name of the scheduler to be used.
        extra_arguments (Optional[Dict[str, Any]): A dictionary containing all other scheduler-specific arguments.
    """

    name: str
    extra_arguments: Optional[Dict[str, Any]] = None

    @validator("name")
    def validate_names(cls, v: str) -> str:
        """
        Validator to ensure each name in `name_list` corresponds to a valid augmentation.
        """
        if v not in AVAILABLE_SCHEDULERS:
            raise ValueError(
                f"Scheduler '{v}' is not implemented. Available schedulers: {list(AVAILABLE_SCHEDULERS.keys())}"
            )
        return v

    @validator("extra_arguments", pre=True, always=True)
    def check_default_extra_args(cls, v: Optional[Dict[str, Any]], values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the 'extra_arguments' by assigning a default empty dictionary if None is provided,
        and also checks necessary keys for different scheduler types.
        """
        if v is None:
            v = {}
        if "name" in values:
            scheduler_name = values["name"]
            required_keys = {
                "cosine_annealing": "T_max",
                "exponential": "gamma",
                "one_cycle": "max_lr",
                "step_lr": "step_size",
            }
            if scheduler_name in required_keys and required_keys[scheduler_name] not in v:
                raise ValueError(f"{required_keys[scheduler_name]} must be provided for {scheduler_name}")
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
