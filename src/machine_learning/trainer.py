# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Literal, Optional, Union

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
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


def create_trainer(
    config: TrainerConfig, callbacks: Optional[List[Callback]], loggers: Optional[List[Logger]]
) -> Trainer:
    """
    Create a PyTorch Lightning Trainer instance from a given configuration.

    Args:
        config (TrainerConfig): The configuration object for the Trainer.
        callbacks (Optional[List[Callback]]): A list of PyTorch Lightning callback to use.
        loggers (Optional[List[Logger]]): A list of PyTorch Lightning loggers to use.

    Returns:
        Trainer: An instance of the PyTorch Lightning Trainer configured as per the provided settings.
    """
    args = {name: args for name, args in config.dict().items() if args is not None}

    _logger.info(f"Creating trainer with the following configuration: {args}")

    trainer = Trainer(logger=loggers, callbacks=callbacks, **args)

    _logger.info("Trainer successfully created.")

    return trainer
