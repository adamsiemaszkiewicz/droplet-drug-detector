# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, validator
from torch.nn import ModuleDict
from torchmetrics import Accuracy, F1Score, JaccardIndex, Metric, Precision, Recall

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


AVAILABLE_METRICS: Dict[str, Type[Metric]] = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
    "jaccard_index": JaccardIndex,
}


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


def create_metric(
    metric_class: Type[Metric], task: str, num_classes: int, arguments: Optional[Dict[str, Any]] = None
) -> Metric:
    """
    Create a metric based on the given name and parameters.

    Args:
        metric_class (Type[Metric]): The metric name.
        task (str): The task type (binary, multiclass, multilabel).
        num_classes (int): The number of classes for classification metrics.
        arguments (Optional[Dict[str, Any]]: Additional arguments specific to the metric.

    Returns:
        Metric: An evaluation metric.
    """
    config = {"task": task, "num_classes": num_classes}
    config.update(arguments or {})

    _logger.info(f"Creating evaluation metric with the following configuration: {config}")

    metric = metric_class(task=task, num_classes=num_classes, **arguments)

    _logger.info(f"Metric {metric_class.__name__} successfully created.")

    return metric


def create_metrics(config: ClassificationMetricsConfig) -> ModuleDict:
    """
    Configure a set of metrics based on a MetricConfig instance.

    Args:
        config: A MetricConfig instance specifying metrics to configure.

    Returns:
        A ModuleDict of configured metrics.
    """
    _logger.info(f"Creating {len(config.name_list)} evaluation metrics")

    metrics = ModuleDict()
    for name, extra_arguments in zip(config.name_list, config.extra_arguments_list):
        metrics[name] = create_metric(
            metric_class=AVAILABLE_METRICS[name],
            task=config.task,
            num_classes=config.num_classes,
            arguments=extra_arguments,
        )

    _logger.info("Metrics configured successfully.")

    return metrics
