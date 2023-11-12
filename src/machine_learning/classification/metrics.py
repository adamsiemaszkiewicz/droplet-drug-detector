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

    @validator("extra_arguments_list")
    def check_lengths(cls, v: List[Dict[str, Any]], values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validates that the 'names' and 'arguments' lists are of the same length.
        """
        if "names" in values and len(values["names"]) != len(v):
            raise ValueError("The number of augmentation names and arguments must be the same.")
        return v


def create_metric(name: str, task: str, num_classes: int, extra_arguments: Dict[str, Any]) -> Metric:
    """
    Create a metric based on the given name and parameters.

    Args:
        name: The metric name.
        task: The task type (binary, multiclass, multilabel).
        num_classes: The number of classes for classification metrics.
        extra_arguments: Additional arguments specific to the metric.

    Returns:
        An instance of a PyTorch Metric.

    Raises:
        ValueError: If the metric name is invalid or required arguments are missing.
    """
    if name not in AVAILABLE_METRICS:
        raise ValueError(f"Metric '{name}' is not available. Available metrics: {list(AVAILABLE_METRICS.keys())}")

    metric_args = {"num_classes": num_classes, "task": task}
    if extra_arguments:
        metric_args.update(extra_arguments)
    metric_class = AVAILABLE_METRICS[name]

    try:
        metric = metric_class(**metric_args)
        _logger.info(f"Metric '{name}' created with arguments: {metric_args}")
    except TypeError as e:
        error_message = f"Incorrect arguments for metric '{name}'. Error: {e}"
        _logger.error(error_message)
        raise ValueError(error_message)

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

    metrics = ModuleDict()
    for name, extra_arguments in zip(config.name_list, config.extra_arguments_list):
        metrics[name] = create_metric(
            name=name, task=config.task, num_classes=config.num_classes, extra_arguments=extra_arguments
        )
    _logger.info("Metrics configured successfully.")

    return metrics
