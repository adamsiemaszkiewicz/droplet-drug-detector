# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Union

from pydantic import Field, validator

from src.common.utils.logger import get_logger
from src.common.utils.validators import FieldValidators
from src.configs.base import (
    BaseCallbacksConfig,
    BaseDataConfig,
    BaseMachineLearningConfig,
    BaseModelConfig,
    BaseTrainerConfig,
)

_logger = get_logger(__name__)


class SampleDataConfig(BaseDataConfig):
    # Attributes
    parameter_1: str = Field()
    parameter_2: int = Field()
    parameter_3: float = Field()
    parameter_4: Path = Field()
    parameter_5: Union[str, List[str]] = Field()
    parameter_6: Union[str, List[int]] = Field()
    parameter_7: Union[str, List[float]] = Field()
    parameter_8: bool = Field()

    # Validators
    _parameter_2a = validator("parameter_2", allow_reuse=True)(FieldValidators.convert_str_to_int)
    _parameter_2b = validator("parameter_2", allow_reuse=True)(FieldValidators.check_if_positive)
    _parameter_3a = validator("parameter_3", allow_reuse=True)(FieldValidators.convert_str_to_float)
    _parameter_3b = validator("parameter_3", allow_reuse=True)(FieldValidators.check_if_positive)
    _parameter_4 = validator("parameter_4", allow_reuse=True)(FieldValidators.convert_str_to_path)
    _parameter_5 = validator("parameter_5", allow_reuse=True)(
        lambda v: FieldValidators.convert_comma_separated_str_to_list(v, str)
    )
    _parameter_6 = validator("parameter_6", allow_reuse=True)(
        lambda v: FieldValidators.convert_comma_separated_str_to_list(v, int)
    )
    _parameter_7 = validator("parameter_7", allow_reuse=True)(
        lambda v: FieldValidators.convert_comma_separated_str_to_list(v, float)
    )
    _parameter_8 = validator("parameter_8", allow_reuse=True)(BaseMachineLearningConfig.convert_str_to_bool)


class SampleModelConfig(BaseModelConfig):
    pass


class SampleTrainerConfig(BaseTrainerConfig):
    pass


class SampleCallbacksConfig(BaseCallbacksConfig):
    pass


class SampleConfig(BaseMachineLearningConfig):
    data: SampleDataConfig
    model: SampleModelConfig
    trainer: SampleTrainerConfig
    callbacks: SampleCallbacksConfig
