# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from pydantic import BaseModel, root_validator

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class TrainerConfig(BaseModel):
    """
    Configuration for creating a PyTorch Lightning Trainer instance.

    This class uses Pydantic for data validation and settings management. Add additional fields
    as required for your specific configuration needs.
    """

    max_epochs: int = 20
    callbacks: Optional[list] = None  # Expect this to be a list of PyTorch Lightning Callback instances
    logger: Optional[list] = None  # Expect this to be a list of PyTorch Lightning Logger instances

    @root_validator(pre=True)
    def check_gpu_availability(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that GPUs are available if specified in the configuration.
        """
        gpus = values.get("gpus", None)
        if gpus:
            # Here you would typically check if the specified number of GPUs are available.
            # For example, using torch.cuda.is_available() and torch.cuda.device_count()
            pass  # Replace with actual validation logic
        return values


def create_trainer(config: TrainerConfig, callbacks: List[Callback], loggers: List[Logger]) -> Trainer:
    """
    Creates a PyTorch Lightning Trainer instance from a given configuration.

    Args:
        config (TrainerConfig): The configuration object for the Trainer.
        callbacks (List[Callback]): A list of PyTorch Lightning callback to use.
        loggers (List[Logger]): A list of PyTorch Lightning loggers to use.

    Returns:
        Trainer: An instance of the PyTorch Lightning Trainer configured as per the provided settings.
    """
    trainer = Trainer(max_epochs=config.max_epochs, logger=loggers, callbacks=callbacks)

    return trainer
