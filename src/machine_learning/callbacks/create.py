# -*- coding: utf-8 -*-
from typing import List

from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.common.utils.logger import get_logger
from src.machine_learning.callbacks.config import CallbacksConfig

_logger = get_logger(__name__)


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
