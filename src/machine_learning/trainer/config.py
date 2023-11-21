# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import validator

from src.common.utils.logger import get_logger
from src.configs.base import BaseTrainerConfig

_logger = get_logger(__name__)


class TrainerConfig(BaseTrainerConfig):
    """
    Configuration for creating a PyTorch Lightning trainer.

    Attributes:
        max_epochs (int): The maximum number of epochs to train for.
        precision (Optional[Literal["16", "32", "64"]): The precision to use for training.
        accumulate_grad_batches (Optional[int]): The number of batches to accumulate gradients for.
        accelerator (Optional[str]): The accelerator to use for training.
        num_devices (Optional[int]): The number of devices to use for training.
        default_root_dir (Optional[Union[str, Path]]): The default root directory for storing logs and checkpoints.
        fast_dev_run (bool): Test if training code run without errors (for debugging purposes only)
        overfit_batches (Union[float, int]): Amount of data to use for overfitting
                                             0.0-1.0 as percentage or integer number of batches
                                             Defaults to 0.0 which uses no overfitting)
    """

    max_epochs: int
    precision: Optional[Literal["16", "32", "64"]] = None
    accumulate_grad_batches: Optional[int] = None
    accelerator: Optional[str] = None
    num_devices: Optional[int] = None
    default_root_dir: Optional[Union[str, Path]] = None

    # Debugging
    fast_dev_run: bool = False
    overfit_batches: float = 0.0

    @validator("max_epochs")
    def validate_positive_integer(cls, v: int) -> int:
        """
        Validates if the provided value is a positive integer.
        """
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"The value {v} must be a positive integer.")
        return v
