# -*- coding: utf-8 -*-
from typing import Optional

from src.aml.components.regression_training.data import RegressionDataConfig
from src.common.utils.logger import get_logger
from src.configs.base import MachineLearningConfig
from src.machine_learning.augmentations.config import AugmentationsConfig
from src.machine_learning.callbacks.config import CallbacksConfig
from src.machine_learning.loggers.config import LoggersConfig
from src.machine_learning.loss_functions.config import RegressionLossFunctionConfig
from src.machine_learning.models.config import RegressionModelConfig
from src.machine_learning.optimizer.config import OptimizerConfig
from src.machine_learning.preprocessing.config import PreprocessingConfig
from src.machine_learning.scheduler.config import SchedulerConfig
from src.machine_learning.trainer.config import TrainerConfig

_logger = get_logger(__name__)


class RegressionConfig(MachineLearningConfig):
    data: RegressionDataConfig
    preprocessing: Optional[PreprocessingConfig] = None
    model: RegressionModelConfig
    loss_function: RegressionLossFunctionConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    augmentations: Optional[AugmentationsConfig] = None
    callbacks: Optional[CallbacksConfig] = None
    loggers: Optional[LoggersConfig] = None
    trainer: TrainerConfig

    seed: int
