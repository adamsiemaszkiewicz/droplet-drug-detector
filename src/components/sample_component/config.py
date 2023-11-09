# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List

from pydantic import Field, validator

from src.common.configs.base import BaseCallbacksConfig, BaseConfig, BaseDataConfig, BaseModelConfig, BaseTrainerConfig
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class SampleDataConfig(BaseDataConfig):
    parameter_1: str = Field(default="data_param1_default", description="Sample parameter for data section.")
    parameter_2: int = Field(default=1, description="Sample parameter for data section.")
    parameter_3: float = Field(default=1.0, description="Sample parameter for data section (float).")
    parameter_4: Path = Field(default=Path("."), description="Sample path parameter for data section.")
    parameter_5: List[str] = Field(
        default=["data,param6,default"], description="Sample list parameter for data section."
    )
    parameter_6: bool = Field(default=True, description="Sample parameter for data section (boolean).")

    # Validators
    validator("parameter_2", allow_reuse=True)(BaseConfig.convert_str_to_positive_int)
    validator("parameter_3", allow_reuse=True)(BaseConfig.convert_str_to_positive_float)
    validator("parameter_4", allow_reuse=True)(BaseConfig.convert_str_to_path)
    validator("parameter_5", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_strings)
    validator("parameter_6", allow_reuse=True)(BaseConfig.convert_str_to_bool)


class SampleModelConfig(BaseModelConfig):
    parameter_1: str = Field(default="data_param1_default", description="Sample parameter for data section.")
    parameter_2: int = Field(default=1, description="Sample parameter for data section.")
    parameter_3: float = Field(default=1.0, description="Sample parameter for data section (float).")
    parameter_4: Path = Field(default=Path("."), description="Sample path parameter for data section.")
    parameter_5: List[str] = Field(
        default=["data,param6,default"], description="Sample list parameter for data section."
    )
    parameter_6: bool = Field(default=True, description="Sample parameter for data section (boolean).")

    # Validators
    validator("parameter_2", allow_reuse=True)(BaseConfig.convert_str_to_positive_int)
    validator("parameter_3", allow_reuse=True)(BaseConfig.convert_str_to_positive_float)
    validator("parameter_4", allow_reuse=True)(BaseConfig.convert_str_to_path)
    validator("parameter_5", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_strings)
    validator("parameter_6", allow_reuse=True)(BaseConfig.convert_str_to_bool)


class SampleTrainerConfig(BaseTrainerConfig):
    parameter_1: str = Field(default="data_param1_default", description="Sample parameter for data section.")
    parameter_2: int = Field(default=1, description="Sample parameter for data section.")
    parameter_3: float = Field(default=1.0, description="Sample parameter for data section (float).")
    parameter_4: Path = Field(default=Path("."), description="Sample path parameter for data section.")
    parameter_5: List[str] = Field(
        default=["data,param6,default"], description="Sample list parameter for data section."
    )
    parameter_6: bool = Field(default=True, description="Sample parameter for data section (boolean).")

    # Validators
    validator("parameter_2", allow_reuse=True)(BaseConfig.convert_str_to_positive_int)
    validator("parameter_3", allow_reuse=True)(BaseConfig.convert_str_to_positive_float)
    validator("parameter_4", allow_reuse=True)(BaseConfig.convert_str_to_path)
    validator("parameter_5", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_strings)
    validator("parameter_6", allow_reuse=True)(BaseConfig.convert_str_to_bool)


class SampleCallbacksConfig(BaseCallbacksConfig):
    parameter_1: str = Field(default="data_param1_default", description="Sample parameter for data section.")
    parameter_2: int = Field(default=1, description="Sample parameter for data section.")
    parameter_3: float = Field(default=1.0, description="Sample parameter for data section (float).")
    parameter_4: Path = Field(default=Path("."), description="Sample path parameter for data section.")
    parameter_5: List[str] = Field(
        default=["data,param6,default"], description="Sample list parameter for data section."
    )
    parameter_6: bool = Field(default=True, description="Sample parameter for data section (boolean).")

    # Validators
    validator("parameter_2", allow_reuse=True)(BaseConfig.convert_str_to_positive_int)
    validator("parameter_3", allow_reuse=True)(BaseConfig.convert_str_to_positive_float)
    validator("parameter_4", allow_reuse=True)(BaseConfig.convert_str_to_path)
    validator("parameter_5", allow_reuse=True)(BaseConfig.convert_comma_separated_str_to_list_of_strings)
    validator("parameter_6", allow_reuse=True)(BaseConfig.convert_str_to_bool)


class SampleConfig(BaseConfig):
    data: SampleDataConfig = SampleDataConfig()
    model: SampleModelConfig = SampleModelConfig()
    trainer: SampleTrainerConfig = SampleTrainerConfig()
    callbacks: SampleCallbacksConfig = SampleCallbacksConfig()
