# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type, Union

from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pydantic import BaseModel, validator

from src.common.utils.logger import get_logger
from src.configs.base import BaseCallbacksConfig

_logger = get_logger(__name__)

AVAILABLE_CALLBACKS: Dict[str, Type[Callback]] = {
    "early_stopping": EarlyStopping,
    "model_checkpoint": ModelCheckpoint,
    "learning_rate_monitor": LearningRateMonitor,
}


class EarlyStoppingCallbackConfig(BaseModel):
    """
    Configuration settings for the EarlyStopping callback.
    """

    monitor: str
    min_delta: float
    patience: int
    verbose: bool
    mode: Literal["min", "max"]


class ModelCheckpointCallbackConfig(BaseModel):
    """
    Configuration settings for the ModelCheckpoint callback.
    """

    dirpath: Path
    monitor: str
    filename: str
    save_top_k: int
    mode: Literal["min", "max"]
    verbose: bool

    @validator("dirpath", pre=True)
    def ensure_path_is_path(cls, v: Union[str, Path]) -> Path:
        """
        Ensures that paths are of type pathlib.Path.
        """
        if not isinstance(v, Path):
            return Path(v)
        return v


class LearningRateMonitorConfig(BaseModel):
    """
    Configuration settings for LearningRateMonitor callback.
    """

    log_momentum: bool = True
    log_weight_decay: bool = True


class CallbacksConfig(BaseCallbacksConfig):
    """
    Configuration for creating a list of callbacks based on their names and configurations.
    """

    early_stopping: Optional[EarlyStoppingCallbackConfig] = None
    model_checkpoint: Optional[ModelCheckpointCallbackConfig] = None
    learning_rate_monitor: Optional[LearningRateMonitorConfig] = None


def create_callbacks(config: CallbacksConfig) -> List[Callback]:
    """
    Creates a list of callback instances from a CallbacksConfig instance.

    Args:
        config (CallbacksConfig): The configuration object containing callback settings.

    Returns:
        List[Callback]: A list of callback instances as per the configuration.
    """
    callbacks = []

    if config.early_stopping:
        _logger.info("Creating EarlyStopping callback instance.")
        callbacks.append(
            EarlyStopping(
                monitor=config.early_stopping.monitor,
                min_delta=config.early_stopping.min_delta,
                patience=config.early_stopping.patience,
                verbose=config.early_stopping.verbose,
                mode=config.early_stopping.mode,
            )
        )

    if config.model_checkpoint:
        _logger.info("Creating ModelCheckpoint callback instance.")
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.model_checkpoint.dirpath,
                monitor=config.model_checkpoint.monitor,
                filename=config.model_checkpoint.filename,
                save_top_k=config.model_checkpoint.save_top_k,
                mode=config.model_checkpoint.mode,
                verbose=config.model_checkpoint.verbose,
            )
        )

    if config.learning_rate_monitor:
        _logger.info("Creating LearningRateMonitor callback instance.")
        callbacks.append(
            LearningRateMonitor(
                log_momentum=config.learning_rate_monitor.log_momentum,
                log_weight_decay=config.learning_rate_monitor.log_weight_decay,
            )
        )

    _logger.info("Callback instances successfully created.")

    return callbacks
