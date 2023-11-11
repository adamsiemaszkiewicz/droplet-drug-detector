# -*- coding: utf-8 -*-
from typing import Any, Dict

from pydantic import BaseModel
from timm.loss import (
    AsymmetricLossMultiLabel,
    AsymmetricLossSingleLabel,
    BinaryCrossEntropy,
    JsdCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from torch.nn import CrossEntropyLoss, Module


class LossConfig(BaseModel):
    """
    Configuration for loss function creation.
    """

    loss_name: str
    loss_args: Dict[str, Any]


LOSS_FUNCTIONS = {
    "asymmetric_loss_multi_label": AsymmetricLossMultiLabel,
    "asymmetric_loss_single_label": AsymmetricLossSingleLabel,
    "binary_cross_entropy": BinaryCrossEntropy,
    "cross_entropy_loss": CrossEntropyLoss,
    "jsd_cross_entropy": JsdCrossEntropy,
    "label_smoothing": LabelSmoothingCrossEntropy,
    "soft_target": SoftTargetCrossEntropy,
}


def create_loss_function(config: LossConfig) -> Module:
    """
    Create a loss function based on the configuration.

    Args:
        config (LossConfig): Configuration object specifying the name and arguments of the loss.

    Returns:
        Module: A PyTorch loss function module.

    Raises:
        NotImplementedError: If the loss_name is not recognized.
    """
    loss_constructor = LOSS_FUNCTIONS.get(config.loss_name)
    if not loss_constructor:
        raise NotImplementedError(f"Augmentation {config.loss_name} is not implemented")

    loss_function = loss_constructor(**config.loss_args)

    return loss_function
