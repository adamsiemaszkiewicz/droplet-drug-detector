# -*- coding: utf-8 -*-
from typing import List, Optional

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger

from src.common.utils.logger import get_logger
from src.machine_learning.trainer.config import TrainerConfig

_logger = get_logger(__name__)


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
