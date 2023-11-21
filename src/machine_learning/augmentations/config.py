# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

from pydantic import validator

from src.common.utils.logger import get_logger
from src.configs.base import BaseAugmentationsConfig
from src.machine_learning.augmentations.types import AVAILABLE_AUGMENTATIONS, REQUIRED_ARGUMENTS

_logger = get_logger(__name__)


class AugmentationsConfig(BaseAugmentationsConfig):
    """
    Configuration for creating a sequence of augmentations.
    """

    name_list: List[str]
    extra_arguments_list: Optional[List[Dict[str, Any]]] = None

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Validates if all augmentation names are implemented.
        """
        if v not in AVAILABLE_AUGMENTATIONS:
            raise ValueError(
                f"Augmentation '{v}' is not implemented.\n"
                f"Available augmentations: {list(AVAILABLE_AUGMENTATIONS.keys())}"
            )
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

    @validator("extra_arguments_list", always=True)
    def validate_required_augmentation_arguments(
        cls, extra_args_list: List[Optional[Dict[str, Any]]], values: Dict[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Validates if the required arguments for each chosen augmentation are provided.
        """
        name_list = values.get("name_list", [])
        if not extra_args_list:
            extra_args_list = [{} for _ in name_list]

        for name, extra_args in zip(name_list, extra_args_list):
            if name in REQUIRED_ARGUMENTS:
                required_arg = REQUIRED_ARGUMENTS[name]
                if extra_args is None or required_arg not in extra_args:
                    raise ValueError(f"Required argument '{required_arg}' for augmentation '{name}' is missing.")

        return extra_args_list
