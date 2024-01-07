# -*- coding: utf-8 -*-
from typing import Any, Dict

from pydantic import BaseModel, validator

from src.common.utils.logger import get_logger
from src.machine_learning.scheduler.types import AVAILABLE_SCHEDULERS, REQUIRED_ARGUMENTS

_logger = get_logger(__name__)


class SchedulerConfig(BaseModel):
    """
    Configuration for creating a learning rate scheduler.

    Attrs:
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
