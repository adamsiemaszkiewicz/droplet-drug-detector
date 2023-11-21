# -*- coding: utf-8 -*-
from typing import Dict, Type

from torch.optim import SGD, Adam, Optimizer

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_OPTIMIZERS: Dict[str, Type[Optimizer]] = {
    "adam": Adam,
    "sgd": SGD,
}
