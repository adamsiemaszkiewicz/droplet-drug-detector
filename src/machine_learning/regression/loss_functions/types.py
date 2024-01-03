# -*- coding: utf-8 -*-
from typing import Dict, Type

from torch.nn import L1Loss, Module, MSELoss, SmoothL1Loss

AVAILABLE_LOSS_FUNCTIONS: Dict[str, Type[Module]] = {
    "mean_squared_error": MSELoss,
    "mean_absolute_error": L1Loss,
    "smooth_l1_loss": SmoothL1Loss,
}
