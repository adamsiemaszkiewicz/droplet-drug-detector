# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Literal, Type

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
        names: A list of strings indicating the names of the metrics to be used.
        tasks: A list of tasks corresponding to each metric.
        num_classes: The number of classes, applicable to all metrics.
        extra_arguments: A list of dictionaries, each containing metric-specific arguments.

    """

    names: List[str]
    tasks: List[Literal["binary", "multiclass", "multilabel"]]
    num_classes: int
    extra_arguments: List[Dict[str, Any]]

    @validator
    def validate_configs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        names, tasks, extra_args = (values.get("names"), values.get("tasks"), values.get("extra_arguments"))
        if not (names and tasks and extra_args and len(names) == len(tasks) == len(extra_args)):
            raise ValueError("Length of names, tasks, and extra_arguments lists must be the same.")
        return values


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

    metric_args = {**extra_arguments, "num_classes": num_classes, "task": task}
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
    metrics = ModuleDict()
    for name, task, extra_args in zip(config.names, config.tasks, config.extra_arguments):
        metrics[name] = create_metric(name, task, config.num_classes, extra_args)
    _logger.info("Metrics configured successfully.")
    return metrics
