# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, Dict, Type, Union

from pydantic import BaseModel
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
