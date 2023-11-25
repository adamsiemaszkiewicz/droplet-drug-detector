# -*- coding: utf-8 -*-
from typing import Dict, Type

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
