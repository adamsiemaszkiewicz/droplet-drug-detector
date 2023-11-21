# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import validator

from src.common.utils.logger import get_logger
from src.configs.base import BaseLoggersConfig
from src.machine_learning.loggers.types import AVAILABLE_LOGGERS

_logger = get_logger(__name__)


class LoggersConfig(BaseLoggersConfig):
    """
    Configuration for creating a list of loggers.
    """

    name_list: List[str]
    save_dir: Path
    extra_arguments_list: List[Dict[str, Any]] = []

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Validates if all metric names are implemented.
        """
        if v not in AVAILABLE_LOGGERS:
            raise ValueError(f"Logger '{v}' is not implemented. Available loggers: {list(AVAILABLE_LOGGERS.keys())}")
        return v

    @validator("extra_arguments_list", pre=True, always=True)
    def default_extra_arguments(
        cls, v: List[Optional[Dict[str, Any]]], values: Dict[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Ensures a correct length of `extra_arguments_list` if none are provided.
        """
        if not v:
            name_list = values.get("name_list", [])
            return [{} for _ in name_list]
        return v

    @validator("extra_arguments_list")
    def validate_number_of_extra_arguments(
        cls, v: List[Dict[str, Any]], values: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Ensures that all required arguments are provided.
        """
        name_list = values.get("name_list")
        if name_list is not None and len(v) != len(name_list):
            raise ValueError(
                f"The number of extra arguments ({len(v)}) does not match the number of loggers ({len(name_list)})."
            )
        return v

    @validator("extra_arguments_list", always=True, each_item=True)
    def validate_missing_extra_arguments(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replaces missing extra arguments with empty dictionaries
        """
        if v is None:
            return {}
        return v

    @validator("save_dir", pre=True)
    def ensure_path_is_path(cls, v: Union[str, Path]) -> Path:
        """
        Ensures that paths are of type pathlib.Path.
        """
        if not isinstance(v, Path):
            return Path(v)
        return v
