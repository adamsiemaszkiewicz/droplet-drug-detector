# -*- coding: utf-8 -*-
from pydantic import BaseSettings

from src.common.machine_learning.augmentations import AugmentationsConfig
from src.common.machine_learning.classification.loss_functions import ClassificationLossFunctionConfig
from src.common.machine_learning.classification.metrics import ClassificationMetricsConfig
from src.common.machine_learning.classification.models import ClassificationModelConfig
from src.common.machine_learning.optimizer import OptimizerConfig
from src.common.machine_learning.scheduler import SchedulerConfig


class ClassificationConfig(BaseSettings):
    model: ClassificationModelConfig
    loss_function: ClassificationLossFunctionConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    metrics: ClassificationMetricsConfig
    augmentations: AugmentationsConfig
