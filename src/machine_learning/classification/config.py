# -*- coding: utf-8 -*-
from typing import Optional

from pydantic import BaseSettings

from src.machine_learning.augmentations import AugmentationsConfig
from src.machine_learning.callbacks import CallbacksConfig
from src.machine_learning.classification.loss_functions import ClassificationLossFunctionConfig
from src.machine_learning.classification.metrics import ClassificationMetricsConfig
from src.machine_learning.classification.models import ClassificationModelConfig
from src.machine_learning.loggers import LoggersConfig
from src.machine_learning.optimizer import OptimizerConfig
from src.machine_learning.scheduler import SchedulerConfig
from src.machine_learning.trainer import TrainerConfig


class ClassificationConfig(BaseSettings):
    model: ClassificationModelConfig
    loss_function: ClassificationLossFunctionConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    metrics: ClassificationMetricsConfig
    augmentations: Optional[AugmentationsConfig] = None
    callbacks: Optional[CallbacksConfig] = None
    loggers: Optional[LoggersConfig] = None
    trainer: TrainerConfig
