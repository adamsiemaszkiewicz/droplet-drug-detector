# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Type

from torch.nn import ModuleDict
from torchmetrics import Metric

from src.common.utils.logger import get_logger
from src.machine_learning.classification.metrics.config import ClassificationMetricsConfig
from src.machine_learning.classification.metrics.types import AVAILABLE_METRICS

_logger = get_logger(__name__)


def create_metric(
    metric_class: Type[Metric], task: str, num_classes: int, arguments: Optional[Dict[str, Any]] = None
) -> Metric:
    """
    Create a metric based on the configuration.

    Args:
        metric_class (Type[Metric]): The metric class.
        task (str): The task type (binary, multiclass, multilabel).
        num_classes (int): The number of classes for classification metrics.
        arguments (Optional[Dict[str, Any]]: Additional arguments specific to the metric.

    Returns:
        Metric: An evaluation metric.
    """
    config = {"task": task, "num_classes": num_classes}
    config.update(arguments or {})

    metric = metric_class(task=task, num_classes=num_classes, **(arguments or {}))

    _logger.info(f"Created metric '{metric_class.__name__}' with the following configuration: {config}")

    return metric


def create_metrics(config: ClassificationMetricsConfig) -> ModuleDict:
    """
    Configure a set of metrics based on the configuration.

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
