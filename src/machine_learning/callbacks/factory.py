# -*- coding: utf-8 -*-
from typing import List

from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.common.utils.logger import get_logger
from src.machine_learning.callbacks.config import CallbacksConfig
from src.machine_learning.callbacks.confusion_matrix import ConfusionMatrixCallback
from src.machine_learning.callbacks.learning_curve import LearningCurveCallback
from src.machine_learning.callbacks.misclassification import MisclassificationCallback

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

    if config.learning_curve_logger:
        _logger.info("Creating LearningCurveCallback callback instance.")
        callbacks.append(
            LearningCurveCallback(
                save_dir=config.learning_curve_logger.save_dir,
                log_loss=config.learning_curve_logger.log_loss,
                log_metrics=config.learning_curve_logger.log_metrics,
            )
        )

    if config.confusion_matrix_logger:
        _logger.info("Creating ConfusionMatrixCallback callback instance.")
        callbacks.append(
            ConfusionMatrixCallback(
                save_dir=config.confusion_matrix_logger.save_dir,
                class_dict=config.confusion_matrix_logger.class_dict,
                task=config.confusion_matrix_logger.task_type,
                log_train=config.confusion_matrix_logger.log_train,
                log_val=config.confusion_matrix_logger.log_val,
                log_test=config.confusion_matrix_logger.log_test,
            )
        )

    if config.misclassification_logger:
        _logger.info("Creating MisclassificationCallback callback instance.")
        callbacks.append(
            MisclassificationCallback(
                save_dir=config.misclassification_logger.save_dir,
                log_train=config.misclassification_logger.log_train,
                log_val=config.misclassification_logger.log_val,
                log_test=config.misclassification_logger.log_test,
            )
        )

    _logger.info("Callback instances successfully created.")

    return callbacks
