# -*- coding: utf-8 -*-
from typing import Dict, Type

from lightning.pytorch.loggers import CSVLogger, Logger, MLFlowLogger, TensorBoardLogger

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_LOGGERS: Dict[str, Type[Logger]] = {
    "csv": CSVLogger,
    "mlflow": MLFlowLogger,
    "tensorboard": TensorBoardLogger,
}
