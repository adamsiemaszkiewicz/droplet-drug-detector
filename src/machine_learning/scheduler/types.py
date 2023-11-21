# -*- coding: utf-8 -*-
from typing import Dict, Type

from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LRScheduler, OneCycleLR, StepLR

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_SCHEDULERS: Dict[str, Type[LRScheduler]] = {
    "cosine_annealing": CosineAnnealingLR,
    "exponential": ExponentialLR,
    "one_cycle": OneCycleLR,
    "step_lr": StepLR,
}

REQUIRED_ARGUMENTS: Dict[str, str] = {
    "cosine_annealing": "T_max",
    "exponential": "gamma",
    "one_cycle": "max_lr",
    "step_lr": "step_size",
}
