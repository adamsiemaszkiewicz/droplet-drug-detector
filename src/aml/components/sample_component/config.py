# -*- coding: utf-8 -*-
from typing import Optional

from src.common.utils.logger import get_logger
from src.configs.base import MachineLearningConfig
from src.machine_learning.augmentations.config import AugmentationsConfig
from src.machine_learning.callbacks.config import CallbacksConfig
from src.machine_learning.classification.loss_functions.config import ClassificationLossFunctionConfig
from src.machine_learning.classification.metrics.config import ClassificationMetricsConfig
from src.machine_learning.classification.models.config import ClassificationModelConfig
from src.machine_learning.data import ClassificationDataConfig
from src.machine_learning.loggers.config import LoggersConfig
from src.machine_learning.optimizer.config import OptimizerConfig
from src.machine_learning.preprocessing.config import PreprocessingConfig
from src.machine_learning.scheduler.config import SchedulerConfig
from src.machine_learning.trainer.config import TrainerConfig

_logger = get_logger(__name__)


class ClassificationConfig(MachineLearningConfig):
    data: ClassificationDataConfig
    preprocessing: Optional[PreprocessingConfig] = None
    model: ClassificationModelConfig
    loss_function: ClassificationLossFunctionConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    metrics: ClassificationMetricsConfig
    augmentations: Optional[AugmentationsConfig] = None
    callbacks: Optional[CallbacksConfig] = None
    loggers: Optional[LoggersConfig] = None
    trainer: TrainerConfig

    seed: int
