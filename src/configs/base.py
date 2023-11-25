# -*- coding: utf-8 -*-
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

from src.common.consts.directories import ARTIFACTS_DIR, ROOT_DIR
from src.common.consts.extensions import YAML
from src.common.utils.dtype_converters import path_to_str
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class MachineLearningConfig(BaseModel):
    data: BaseModel
    preprocessing: Optional[BaseModel] = None
    model: BaseModel
    loss_function: BaseModel
    optimizer: BaseModel
    scheduler: Optional[BaseModel] = None
    metrics: BaseModel
    augmentations: Optional[BaseModel] = None
    callbacks: Optional[BaseModel] = None
    loggers: Optional[BaseModel] = None
    trainer: BaseModel

    seed: int

    class Config:
        json_encoders = {
            Path: lambda v: v.as_posix(),
        }

    def __str__(self) -> str:
        """
        Return a string representation of the configuration in JSON format.

        Returns:
            str: A JSON formatted string representation of the configuration.
        """
        return json.dumps(self.dict(), indent=4, default=pydantic_encoder)

    def log_self(self) -> None:
        """
        Log the string representation of the configuration.
        """
        _logger.info(self.__str__())

    def to_yaml(self, path: Optional[Path] = None) -> None:
        """
        Save the configuration to a YAML file.

        Args:
            path (Path): The path where the YAML file will be saved.
        """
        if path:
            _logger.info(f"Saving configuration to: {path.as_posix()}")

        else:
            timestamp = datetime.now().isoformat()
            path = ARTIFACTS_DIR / timestamp / f"config{YAML}"
            _logger.info(f"No save path specified. Saving configuration to a default path: {path.as_posix()}")

        config_dict = self.dict(by_alias=True)
        self._remove_root_dir_from_paths(config_dict)
        self._convert_paths_to_strings(config_dict)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f)

        _logger.info("Successfully exported configuration.")

    @classmethod
    def from_yaml(cls, path: Path) -> "MachineLearningConfig":
        """
        Create a configuration instance from a YAML file.

        Args:
            path (Path): The path to the YAML file.

        Returns:
            ClassificationConfig: A configuration instance.
        """
        with open(path) as f:
            args = yaml.safe_load(f)

        cls._add_root_dir_to_paths(data=args)
        return cls(**args)

    @staticmethod
    def _convert_paths_to_strings(data: Any) -> None:
        """
        Recursively converts Path objects to strings in a data structure.

        Args:
            data (Any): The data structure containing Path objects.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, Path):
                    data[key] = path_to_str(value)
                else:
                    MachineLearningConfig._convert_paths_to_strings(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, Path):
                    data[i] = path_to_str(item)
                else:
                    MachineLearningConfig._convert_paths_to_strings(item)

    @staticmethod
    def _add_root_dir_to_paths(data: Any) -> None:
        """
        Adds ROOT_DIR to paths in a given data structure to make them absolute.

        Args:
            data (Any): The data structure with paths to modify.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    absolute_path = (ROOT_DIR / value).as_posix()
                    data[key] = absolute_path
                else:
                    MachineLearningConfig._add_root_dir_to_paths(data=value)
        elif isinstance(data, list):
            for item in data:
                MachineLearningConfig._add_root_dir_to_paths(item)

    @staticmethod
    def _remove_root_dir_from_paths(data: Any) -> None:
        """
        Converts absolute paths in the configuration to relative paths.

        Args:
            data (Any): The data structure containing paths to modify.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and value.startswith(ROOT_DIR.as_posix()):
                    data[key] = value.replace(ROOT_DIR.as_posix(), "")
                else:
                    MachineLearningConfig._remove_root_dir_from_paths(value)
        elif isinstance(data, list):
            for item in data:
                MachineLearningConfig._remove_root_dir_from_paths(item)
