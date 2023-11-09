# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, Dict, List, Type, Union

from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

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


class BaseTrainerConfig(BaseModel):
    """
    Base trainer configuration class from which all trainer config classes should inherit.
    """

    pass


class BaseCallbacksConfig(BaseModel):
    """
    Base callbacks configuration class from which all callbacks config classes should inherit.
    """

    pass


class BaseConfig(BaseModel):
    """
    Base configuration class that aggregates all individual sections of the configuration.
    It includes data, model, trainer, and callbacks configurations.
    """

    data: BaseDataConfig = BaseDataConfig()
    model: BaseModelConfig = BaseModelConfig()
    trainer: BaseTrainerConfig = BaseTrainerConfig()
    callbacks: BaseCallbacksConfig = BaseCallbacksConfig()

    @staticmethod
    def convert_str_to_path(v: Union[str, Path]) -> Path:
        """
        Convert a string to a positive integer, raising an error if conversion is not possible or the value is negative.
        """
        if isinstance(v, str):
            return Path(v)
        return v

    @staticmethod
    def convert_str_to_positive_int(v: Union[int, str], field: Field) -> int:
        """
        Convert a string to a positive float, raising an error if conversion is not possible or the value is negative.
        """
        if not isinstance(v, int):
            try:
                v = int(v)
            except ValueError:
                raise ValueError(f"{field.name} must be convertible to a positive integer")
        if v < 0:
            raise ValueError(f"{field.name} must be a positive integer")
        return v

    @staticmethod
    def convert_str_to_positive_float(v: Union[float, str], field: Field) -> float:
        """
        Convert a string to a boolean, raising an error if conversion is not possible.
        """
        if not isinstance(v, float):
            try:
                v = float(v)
            except ValueError:
                raise ValueError(f"{field.name} must be convertible to a positive float")
        if v < 0:
            raise ValueError(f"{field.name} must be a positive float")
        return v

    @staticmethod
    def convert_str_to_bool(v: Union[bool, str], field: Field) -> bool:
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
            raise ValueError(f"{field.name} must be a boolean value")

    @staticmethod
    def convert_comma_separated_str_to_list_of_strings(v: Union[str, List[str]]) -> List[str]:
        """
        Convert a comma-separated string to a list of strings.
        """
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        raise ValueError("Input must be a comma-separated string or a list of strings")

    @staticmethod
    def convert_comma_separated_str_to_list_of_integers(v: str) -> List[int]:
        """
        Convert a comma-separated string to a list of integers.
        """
        try:
            return [int(item.strip()) for item in v.split(",")]
        except ValueError as e:
            raise ValueError(f"Each item in the list should be convertible to an integer. Error: {e}")

    @staticmethod
    def convert_comma_separated_str_to_list_of_floats(v: str) -> List[float]:
        """
        Convert a comma-separated string to a list of floats.
        """
        try:
            return [float(item.strip()) for item in v.split(",")]
        except ValueError as e:
            raise ValueError(f"Each item in the list should be convertible to a float. Error: {e}")

    class Config:
        json_encoders = {
            Path: lambda v: v.as_posix(),
        }

    @classmethod
    def from_dict(cls: Type["BaseConfig"], config_dict: Dict[str, Union[Dict[str, Any], Any]]) -> "BaseConfig":
        """
        Create an instance of BaseConfig from a dictionary.

        Args:
            config_dict (Dict[str, Union[Dict[str, Any], Any]]): A dictionary containing the configuration.

        Returns:
            BaseConfig: An instance of BaseConfig with the configuration provided.
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Union[Dict[str, Any], Any]]:
        """
        Convert the BaseConfig instance to a dictionary.

        Returns:
            Dict[str, Union[Dict[str, Any], Any]]: A dictionary representation of the BaseConfig instance.
        """
        return self.dict()

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
