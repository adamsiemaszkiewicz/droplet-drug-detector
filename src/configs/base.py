# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, List, Optional, Type, Union

import yaml
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

from src.common.consts.directories import ROOT_DIR
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class BaseDataConfig(BaseModel):
    """
    Base data configuration class from which all data config classes should inherit.
    """

    pass


class BaseModelConfig(BaseModel):
    """
    Base model configuration class from which all model config classes should inherit.
    """

    pass


class BaseLossFunctionConfig(BaseModel):
    """
    Base loss function configuration class from which all loss function config classes should inherit.
    """

    pass


class BaseOptimizerConfig(BaseModel):
    """
    Base optimizer configuration class from which all optimizer config classes should inherit.
    """

    pass


class BaseSchedulerConfig(BaseModel):
    """
    Base scheduler configuration class from which all scheduler config classes should inherit.
    """

    pass


class BaseMetricsScheduler(BaseModel):
    """
    Base metrics scheduler configuration class from which all metrics scheduler config classes should inherit.
    """

    pass


class BaseAugmentationsConfig(BaseModel):
    """
    Base augmentations configuration class from which all augmentations config classes should inherit.
    """

    pass


class BaseCallbacksConfig(BaseModel):
    """
    Base callbacks configuration class from which all callbacks config classes should inherit.
    """

    pass


class BaseLoggersConfig(BaseModel):
    """
    Base loggers configuration class from which all loggers config classes should inherit.
    """

    pass


class BaseTrainerConfig(BaseModel):
    """
    Base trainer configuration class from which all trainer config classes should inherit.
    """

    pass


class BaseConfig(BaseModel):
    """
    Base configuration class that aggregates all individual sections of the configuration.
    It includes data, model, loss function, optimizer, scheduler, metric, augmentations,
    callbacks, loggers & trainer configurations.
    """

    data: BaseDataConfig
    model: BaseModelConfig
    loss_function: BaseLossFunctionConfig
    optimizer: BaseOptimizerConfig = BaseOptimizerConfig()
    scheduler: Optional[BaseSchedulerConfig] = None
    metrics: BaseMetricsScheduler
    augmentations: Optional[BaseAugmentationsConfig] = None
    callbacks: Optional[BaseCallbacksConfig] = None
    loggers: Optional[BaseLoggersConfig] = None
    trainer: BaseTrainerConfig

    class Config:
        json_encoders = {
            Path: lambda v: v.as_posix(),
        }

    def __str__(self) -> str:
        """
        Return a string representation of the BaseConfig instance in JSON format.

        Returns:
            str: A JSON formatted string representation of the BaseConfig instance.
        """
        return json.dumps(self.dict(), indent=4, default=pydantic_encoder)

    def log_self(self) -> None:
        """
        Log the string representation of the BaseConfig instance.
        """
        _logger.info(self.__str__())

    @classmethod
    def from_yaml(cls, path: Path) -> "BaseConfig":
        """
        Create a BaseConfig instance from a YAML file.

        Args:
            path (Union[str, Path]): The path to the YAML file.

        Returns:
            BaseConfig: A BaseConfig instance.
        """
        with open(path) as f:
            args = yaml.safe_load(f)

        for key, value in args.items():
            if isinstance(value, str) and value.startswith("path:"):
                relative_path = value.split("path:", 1)[1]
                args[key] = str(ROOT_DIR / relative_path)

        return cls(**args)

    @staticmethod
    def convert_str_to_int(value: str) -> int:
        """
        Convert a string to an integer.

        Args:
            value (str): The string to convert.

        Returns:
            int: The converted integer.
        """
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(f"Error converting '{value}' to int: {e}")

    @staticmethod
    def convert_str_to_float(value: str) -> float:
        """
        Convert a string to a float.

        Args:
            value (str): The string to convert.

        Returns:
            float: The converted float.
        """
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Error converting '{value}' to float: {e}")

    @staticmethod
    def convert_str_to_path(v: Union[str, Path]) -> Path:
        """
        Convert a string to a positive integer, raising an error if conversion is not possible or the value is negative.
        """
        if isinstance(v, str):
            return Path(v)
        return v

    @staticmethod
    def convert_str_to_bool(v: Union[bool, str], field: str) -> bool:
        """
        Convert a string to a boolean, raising an error if conversion is not possible.
        """
        if isinstance(v, bool):
            return v
        v = v.lower()
        if v in ("true", "1", "t", "y", "yes"):
            return True
        elif v in ("false", "0", "f", "n", "no"):
            return False
        else:
            raise ValueError(f"{field} must be a boolean value")

    @staticmethod
    def check_if_positive(value: Any) -> Any:
        """
        Check if the provided value is positive.

        Args:
            value (Any): The value to check for positivity.

        Returns:
            Any: The original value if positive.

        Raises:
            ValueError: If the value is not a positive number.
        """
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError("The value must be a positive number.")
        return value

    @staticmethod
    def convert_str_to_positive_int(v: Union[int, str], field: str) -> int:
        """
        Convert a string to a positive float, raising an error if conversion is not possible or the value is negative.
        """
        if not isinstance(v, int):
            try:
                v = int(v)
            except ValueError:
                raise ValueError(f"{field} must be convertible to a positive integer")
        if v < 0:
            raise ValueError(f"{field} must be a positive integer")
        return v

    @staticmethod
    def convert_str_to_positive_float(v: Union[float, str], field: str) -> float:
        """
        Convert a string to a boolean, raising an error if conversion is not possible.
        """
        if not isinstance(v, float):
            try:
                v = float(v)
            except ValueError:
                raise ValueError(f"{field} must be convertible to a positive float")
        if v < 0:
            raise ValueError(f"{field} must be a positive float")
        return v

    @staticmethod
    def convert_comma_separated_str_to_list(v: str, convert_to_type: Type) -> List:
        """
        Convert a comma-separated string to a list of a specified type.

        Args:
            v (str): The input string to convert.
            convert_to_type: The type to convert the list items to. Defaults to str.

        Returns:
            List: A list of the specified type with the converted values.

        Raises:
            ValueError: If conversion of any list item fails.
        """
        try:
            return [convert_to_type(item.strip()) for item in v.split(",") if item.strip()]
        except ValueError as e:
            raise ValueError(f"Error converting values to {convert_to_type.__name__}: {e}")
