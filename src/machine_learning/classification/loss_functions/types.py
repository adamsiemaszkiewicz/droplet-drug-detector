# -*- coding: utf-8 -*-
from typing import Dict, Type

from timm.loss import (
    AsymmetricLossMultiLabel,
    AsymmetricLossSingleLabel,
    BinaryCrossEntropy,
    JsdCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from torch.nn import CrossEntropyLoss, Module

AVAILABLE_LOSS_FUNCTIONS: Dict[str, Type[Module]] = {
    "asymmetric_loss_multi_label": AsymmetricLossMultiLabel,
    "asymmetric_loss_single_label": AsymmetricLossSingleLabel,
    "binary_cross_entropy": BinaryCrossEntropy,
    "cross_entropy_loss": CrossEntropyLoss,
    "jsd_cross_entropy": JsdCrossEntropy,
    "label_smoothing_cross_entropy": LabelSmoothingCrossEntropy,
    "soft_target_cross_entropy": SoftTargetCrossEntropy,
}
