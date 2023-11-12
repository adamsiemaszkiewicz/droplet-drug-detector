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
    Configuration for creating metrics used for model evaluation.

    Attributes:
        namelist: A list of strings indicating the names of the metrics to be used.
        task: A type of classification task
        num_classes: The number of classes, applicable to all metrics.
        extra_arguments_list: A list of dictionaries, each containing metric-specific arguments.

    """

    name_list: List[str]
    task: Literal["binary", "multiclass", "multilabel"]
    num_classes: int
    extra_arguments_list: Optional[List[Dict[str, Any]]] = None

    @validator("extra_arguments_list", pre=True)
    def fill_empty_extra_arguments_list(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pre-validator to ensure `extra_arguments_list` is populated with empty dictionaries if None.
        """
        name_list, extra_arguments_list = values.get("name_list"), values.get("extra_arguments_list")
        if extra_arguments_list is None and name_list is not None:
            extra_arguments_list = [{} for _ in name_list]
        values["extra_arguments_list"] = extra_arguments_list
        return values

    @validator("name_list", "extra_arguments_list")
    def validate_list_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validator to check the length of `name_list` and `extra_arguments_list` are the same.
        """
        name_list, extra_arguments_list = values.get("name_list"), values.get("extra_arguments_list")
        if name_list is not None and extra_arguments_list is not None:
            if len(name_list) != len(extra_arguments_list):
                raise ValueError("The length of 'name_list' and 'extra_arguments_list' must be the same.")
        return values

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Validator to ensure each name in `name_list` corresponds to a valid metric.
        """
        if v not in AVAILABLE_METRICS:
            raise ValueError(
                f"Augmentation '{v}' is not implemented. Available augmentations: {list(AVAILABLE_METRICS.keys())}"
            )
        return v


def create_metric(metric_class: Type[Metric], task: str, num_classes: int, extra_arguments: Dict[str, Any]) -> Metric:
    """
    Create a metric based on the given name and parameters.

    Args:
        metric_class (Type[Metric]): The metric name.
        task: The task type (binary, multiclass, multilabel).
        num_classes: The number of classes for classification metrics.
        extra_arguments: Additional arguments specific to the metric.

    Returns:
        An instance of a PyTorch Metric.

    Raises:
        ValueError: If the metric name is invalid or required arguments are missing.
    """
    try:
        metric = metric_class(task=task, num_classes=num_classes, **extra_arguments)
    except TypeError as e:
        raise ValueError(f"Incorrect arguments for {metric_class.__name__}: {e}")

    return metric


def create_metrics(config: ClassificationMetricsConfig) -> ModuleDict:
    """
    Configure a set of metrics based on a MetricConfig instance.

    Args:
        config: A MetricConfig instance specifying metrics to configure.

    Returns:
        A ModuleDict of configured metrics.
    """
    if config.extra_arguments_list is None:
        raise ValueError("'extra_arguments_list' cannot be None")

    _logger.info(f"Creating module dictionary with the following metrics: {config.name_list}")

    metrics = ModuleDict()
    for name, extra_arguments in zip(config.name_list, config.extra_arguments_list):
        metrics[name] = create_metric(
            metric_class=AVAILABLE_METRICS[name],
            task=config.task,
            num_classes=config.num_classes,
            extra_arguments=extra_arguments,
        )
    _logger.info("Metrics configured successfully.")

    return metrics
