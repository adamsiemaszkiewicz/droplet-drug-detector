# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List

from pydantic import Field, validator

from src.common.configs.base import BaseCallbacksConfig, BaseConfig, BaseDataConfig, BaseModelConfig, BaseTrainerConfig
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class SampleDataConfig(BaseDataConfig):
    parameter_1: str = Field()
    parameter_2: int = Field()
    parameter_3: float = Field()
    parameter_4: Path = Field()
    parameter_5: List[str] = Field()
    parameter_6: List[int] = Field()
    parameter_7: List[float] = Field()
    parameter_8: bool = Field()

    # Validators
    validator("parameter_2", allow_reuse=True)(BaseConfig.convert_str_to_positive_int)
    validator("parameter_3", allow_reuse=True)(BaseConfig.convert_str_to_positive_float)
    validator("parameter_4", allow_reuse=True)(BaseConfig.convert_str_to_path)
    validator("parameter_5", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_strings)
    validator("parameter_6", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_integers)
    validator("parameter_7", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_floats)
    validator("parameter_8", allow_reuse=True)(BaseConfig.convert_str_to_bool)


class SampleModelConfig(BaseModelConfig):
    parameter_1: str = Field()
    parameter_2: int = Field()
    parameter_3: float = Field()
    parameter_4: Path = Field()
    parameter_5: List[str] = Field()
    parameter_6: List[int] = Field()
    parameter_7: List[float] = Field()
    parameter_8: bool = Field()

    # Validators
    validator("parameter_2", allow_reuse=True)(BaseConfig.convert_str_to_positive_int)
    validator("parameter_3", allow_reuse=True)(BaseConfig.convert_str_to_positive_float)
    validator("parameter_4", allow_reuse=True)(BaseConfig.convert_str_to_path)
    validator("parameter_5", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_strings)
    validator("parameter_6", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_integers)
    validator("parameter_7", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_floats)
    validator("parameter_8", allow_reuse=True)(BaseConfig.convert_str_to_bool)


class SampleTrainerConfig(BaseTrainerConfig):
    parameter_1: str = Field()
    parameter_2: int = Field()
    parameter_3: float = Field()
    parameter_4: Path = Field()
    parameter_5: List[str] = Field()
    parameter_6: List[int] = Field()
    parameter_7: List[float] = Field()
    parameter_8: bool = Field()

    # Validators
    validator("parameter_2", allow_reuse=True)(BaseConfig.convert_str_to_positive_int)
    validator("parameter_3", allow_reuse=True)(BaseConfig.convert_str_to_positive_float)
    validator("parameter_4", allow_reuse=True)(BaseConfig.convert_str_to_path)
    validator("parameter_5", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_strings)
    validator("parameter_6", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_integers)
    validator("parameter_7", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_floats)
    validator("parameter_8", allow_reuse=True)(BaseConfig.convert_str_to_bool)


class SampleCallbacksConfig(BaseCallbacksConfig):
    parameter_1: str = Field()
    parameter_2: int = Field()
    parameter_3: float = Field()
    parameter_4: Path = Field()
    parameter_5: List[str] = Field()
    parameter_6: List[int] = Field()
    parameter_7: List[float] = Field()
    parameter_8: bool = Field()

    # Validators
    validator("parameter_2", allow_reuse=True)(BaseConfig.convert_str_to_positive_int)
    validator("parameter_3", allow_reuse=True)(BaseConfig.convert_str_to_positive_float)
    validator("parameter_4", allow_reuse=True)(BaseConfig.convert_str_to_path)
    validator("parameter_5", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_strings)
    validator("parameter_6", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_integers)
    validator("parameter_7", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_floats)
    validator("parameter_8", allow_reuse=True)(BaseConfig.convert_str_to_bool)


class SampleConfig(BaseConfig):
    data: SampleDataConfig
    model: SampleModelConfig
    trainer: SampleTrainerConfig
    callbacks: SampleCallbacksConfig
