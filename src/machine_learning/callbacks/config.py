# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, validator

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


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


class LearningCurveCallbackConfig(BaseModel):
    """
    Configuration settings for LearningCurveCallback callback.
    """

    output_dir: Path
    log_loss: bool
    log_metrics: bool

    @validator("output_dir", pre=True)
    def ensure_path_is_path(cls, v: Union[str, Path]) -> Path:
        """
        Ensures that paths are of type pathlib.Path.
        """
        if not isinstance(v, Path):
            return Path(v)
        return v


class CallbacksConfig(BaseModel):
    """
    Configuration for creating a list of callbacks based on their names and configurations.
    """

    early_stopping: Optional[EarlyStoppingCallbackConfig] = None
    model_checkpoint: Optional[ModelCheckpointCallbackConfig] = None
    learning_rate_monitor: Optional[LearningRateMonitorConfig] = None
    learning_curve_logger: Optional[LearningCurveCallbackConfig] = None
