# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, validator

from src.common.utils.logger import get_logger
from src.machine_learning.classification.metrics.types import AVAILABLE_METRICS

_logger = get_logger(__name__)


class ClassificationMetricsConfig(BaseModel):
    """
    Configuration for creating model evaluation metrics.

    Attributes:
        name_list: A list of strings indicating the names of the metrics to be used.
        task: A type of classification task
        num_classes: The number of classes, applicable to all metrics.
        extra_arguments_list: A list of dictionaries, each containing metric-specific arguments.

    """

    name_list: List[str]
    task: Literal["binary", "multiclass", "multilabel"]
    num_classes: int
    extra_arguments_list: List[Optional[Dict[str, Any]]] = []

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Validates if all metric names are implemented.
        """
        if v not in AVAILABLE_METRICS:
            raise ValueError(f"Metric '{v}' is not implemented. Available metrics: {list(AVAILABLE_METRICS.keys())}")
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
                f"The number of extra arguments ({len(v)}) does not match the number of metrics ({len(name_list)})."
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

    @validator("num_classes")
    def validate_positive_integer(cls, v: int) -> int:
        """
        Validates if the provided value is a positive integer.
        """
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"The value {v} must be a positive integer.")
        return v
