# -*- coding: utf-8 -*-
from typing import Dict, Type

from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_CALLBACKS: Dict[str, Type[Callback]] = {
    "early_stopping": EarlyStopping,
    "model_checkpoint": ModelCheckpoint,
    "learning_rate_monitor": LearningRateMonitor,
}
