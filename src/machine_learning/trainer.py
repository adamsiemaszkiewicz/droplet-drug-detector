# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Literal, Optional, Type, Union

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from pydantic import BaseModel, validator

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class TrainerConfig(BaseModel):
    """
    Configuration for creating a PyTorch Lightning trainer.

    Attributes:
        max_epochs: The maximum number of epochs to train for.
        precision: The precision to use for training.
        accumulate_grad_batches: The number of batches to accumulate gradients for.
        default_root_dir: The default root directory for storing logs and checkpoints.
        fast_dev_run: Enable for debugging purposes only.
        overfit_batches: Amount of data to use for overfitting (0.0-1.0 as percentage or integer number of batches).
        callbacks: A list of PyTorch Lightning callbacks to use.
        logger: A list of PyTorch Lightning loggers to use.
    """

    max_epochs: int
    precision: Literal["16", "32", "64"] = "32"
    accumulate_grad_batches: int = 1
    default_root_dir: Optional[Union[str, Path]] = None

    fast_dev_run: bool = False
    overfit_batches: float = 0.0

    inference_mode: bool = False

    callbacks: Optional[List[Type[Callback]]] = None
    logger: Optional[List[Type[Logger]]] = None

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
    _logger.info(f"Creating trainer with the following configuration: {config.dict()}")

    trainer = Trainer(max_epochs=config.max_epochs, logger=loggers, callbacks=callbacks)

    _logger.info("Trainer successfully created.")

    return trainer
