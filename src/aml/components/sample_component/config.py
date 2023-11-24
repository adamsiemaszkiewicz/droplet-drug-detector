# -*- coding: utf-8 -*-
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

from src.common.consts.directories import CONFIGS_LOG_DIR, ROOT_DIR
from src.common.consts.extensions import YAML
from src.common.utils.dtype_converters import path_to_str
from src.common.utils.logger import get_logger
from src.machine_learning.augmentations.config import AugmentationsConfig
from src.machine_learning.callbacks.config import CallbacksConfig
from src.machine_learning.classification.loss_functions import ClassificationLossFunctionConfig
from src.machine_learning.classification.metrics import ClassificationMetricsConfig
from src.machine_learning.classification.models import ClassificationModelConfig
from src.machine_learning.data import ClassificationDataConfig
from src.machine_learning.loggers.config import LoggersConfig
from src.machine_learning.optimizer.config import OptimizerConfig
from src.machine_learning.preprocessing.config import PreprocessingConfig
from src.machine_learning.scheduler.config import SchedulerConfig
from src.machine_learning.trainer.config import TrainerConfig

_logger = get_logger(__name__)


class ClassificationConfig(BaseModel):
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

    def to_yaml(self, path: Optional[Path] = None) -> None:
        """
        Save the ClassificationConfig instance to a YAML file.

        Args:
            path (Path): The path where the YAML file will be saved.
        """
        if path:
            _logger.info(f"Saving configuration to: {path.as_posix()}")

        else:
            timestamp = datetime.now().isoformat()
            path = CONFIGS_LOG_DIR / f"{timestamp}{YAML}"
            _logger.info(f"No save path specified. Saving configuration to a default path: {path.as_posix()}")

        config_dict = self.dict(by_alias=True)
        self._convert_paths_to_strings(config_dict)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f)

        _logger.info("Successfully exported configuration.")

    @classmethod
    def from_yaml(cls, path: Path) -> "ClassificationConfig":
        """
        Create a ClassificationMachineLearningConfig instance from a YAML file.

        Args:
            path (Path): The path to the YAML file.

        Returns:
            ClassificationConfig: A ClassificationMachineLearningConfig instance.
        """
        with open(path) as f:
            args = yaml.safe_load(f)

        cls._process_yaml_values(data=args)
        return cls(**args)

    @staticmethod
    def _convert_paths_to_strings(data: Any) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, Path):
                    data[key] = path_to_str(value)
                else:
                    ClassificationConfig._convert_paths_to_strings(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, Path):
                    data[i] = path_to_str(item)
                else:
                    ClassificationConfig._convert_paths_to_strings(item)

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
                    ClassificationConfig._process_yaml_values(data=value)
        elif isinstance(data, list):
            for item in data:
                ClassificationConfig._process_yaml_values(item)
