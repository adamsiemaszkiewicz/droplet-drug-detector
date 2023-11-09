# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, Dict, List, Type, Union

from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class BaseDataConfig(BaseModel):
    pass


class BaseModelConfig(BaseModel):
    pass


class BaseTrainerConfig(BaseModel):
    pass


class BaseCallbacksConfig(BaseModel):
    pass


class BaseConfig(BaseModel):
    data: BaseDataConfig = BaseDataConfig()
    model: BaseModelConfig = BaseModelConfig()
    trainer: BaseTrainerConfig = BaseTrainerConfig()
    callbacks: BaseCallbacksConfig = BaseCallbacksConfig()

    @staticmethod
    def convert_str_to_path(v: Union[str, Path]) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v

    @staticmethod
    def convert_str_to_positive_int(v: Union[int, str], field: Field) -> int:
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
        if not isinstance(v, float):
            try:
                v = float(v)
            except ValueError:
                raise ValueError(f"{field.name} must be convertible to a positive float")
        if v < 0:
            raise ValueError(f"{field.name} must be a positive float")
        return v

    @staticmethod
    def convert_to_bool(v: Union[bool, str], field: Field) -> bool:
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
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        raise ValueError("Input must be a comma-separated string or a list of strings")

    @staticmethod
    def convert_comma_separated_str_to_list_of_integers(v: str) -> List[int]:
        try:
            return [int(item.strip()) for item in v.split(",")]
        except ValueError as e:
            raise ValueError(f"Each item in the list should be convertible to an integer. Error: {e}")

    @staticmethod
    def convert_comma_separated_str_to_list_of_floats(v: str) -> List[float]:
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
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Union[Dict[str, Any], Any]]:
        return self.dict()

    def __str__(self) -> str:
        return json.dumps(self.dict(), indent=4, default=pydantic_encoder)

    def log_self(self) -> None:
        _logger.info(self.__str__())
