# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

from src.common.consts.directories import ROOT_DIR
from src.common.utils.logger import get_logger
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


class ClassificationMachineLearningConfig(BaseModel):
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

    class Config:
        json_encoders = {
            Path: lambda v: v.as_posix(),
        }

    def __str__(self) -> str:
        """
        Return a string representation of the BaseMachineLearningConfig instance in JSON format.

        Returns:
            str: A JSON formatted string representation of the BaseMachineLearningConfig instance.
        """
        return json.dumps(self.dict(), indent=4, default=pydantic_encoder)

    def log_self(self) -> None:
        """
        Log the string representation of the BaseMachineLearningConfig instance.
        """
        _logger.info(self.__str__())

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ClassificationMachineLearningConfig":
        """
        Create a ClassificationMachineLearningConfig instance from a YAML file.

        Args:
            path (Union[str, Path]): The path to the YAML file.

        Returns:
            ClassificationMachineLearningConfig: A ClassificationMachineLearningConfig instance.
        """
        with open(path) as f:
            args = yaml.safe_load(f)

        cls._process_yaml_values(data=args)
        return cls(**args)

    @staticmethod
    def _process_yaml_values(data: Any) -> None:
        """
        Recursively process values in the YAML structure.

        Args:
            data (Any): The current level of the YAML structure.
        """
        path_prefix = "path://"  # Prefix to indicate that the value is a path
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and value.startswith(path_prefix):
                    relative_path = value.split(path_prefix, 1)[1]  # Remove leading slashes
                    absolute_path = (ROOT_DIR / relative_path).as_posix()
                    data[key] = absolute_path
                else:
                    ClassificationMachineLearningConfig._process_yaml_values(data=value)
        elif isinstance(data, list):
            for item in data:
                ClassificationMachineLearningConfig._process_yaml_values(item)
