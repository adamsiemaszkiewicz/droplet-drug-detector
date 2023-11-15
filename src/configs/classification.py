# -*- coding: utf-8 -*-
from typing import Optional

from src.common.utils.logger import get_logger
from src.configs.base import BaseMachineLearningConfig
from src.machine_learning.augmentations import AugmentationsConfig
from src.machine_learning.callbacks import CallbacksConfig
from src.machine_learning.classification.loss_functions import ClassificationLossFunctionConfig
from src.machine_learning.classification.metrics import ClassificationMetricsConfig
from src.machine_learning.classification.models import ClassificationModelConfig
from src.machine_learning.data import ClassificationDataConfig
from src.machine_learning.loggers import LoggersConfig
from src.machine_learning.optimizer import OptimizerConfig
from src.machine_learning.preprocessing import PreprocessingConfig
from src.machine_learning.scheduler import SchedulerConfig
from src.machine_learning.trainer import TrainerConfig

_logger = get_logger(__name__)


class ClassificationMachineLearningConfig(BaseMachineLearningConfig):
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
