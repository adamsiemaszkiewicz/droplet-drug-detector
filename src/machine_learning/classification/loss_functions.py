# -*- coding: utf-8 -*-
from typing import Any, Dict, Type

from pydantic import BaseModel, validator
from timm.loss import (
    AsymmetricLossMultiLabel,
    AsymmetricLossSingleLabel,
    BinaryCrossEntropy,
    JsdCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from torch.nn import CrossEntropyLoss, Module

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_LOSS_FUNCTIONS: Dict[str, Type[Module]] = {
    "asymmetric_loss_multi_label": AsymmetricLossMultiLabel,
    "asymmetric_loss_single_label": AsymmetricLossSingleLabel,
    "binary_cross_entropy": BinaryCrossEntropy,
    "cross_entropy_loss": CrossEntropyLoss,
    "jsd_cross_entropy": JsdCrossEntropy,
    "label_smoothing_cross_entropy": LabelSmoothingCrossEntropy,
    "soft_target_cross_entropy": SoftTargetCrossEntropy,
}


class ClassificationLossFunctionConfig(BaseModel):
    """
    Configuration for creating a loss function.

    Attributes:
        name: The name of the loss function.
        extra_arguments: A dictionary containing all other loss function-specific arguments.
    """

    name: str
    extra_arguments: Dict[str, Any] = {}

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validates if the loss function is implemented.
        """
        if v not in AVAILABLE_LOSS_FUNCTIONS:
            raise ValueError(
                f"Loss function '{v}' is not implemented.\n"
                f"Available loss functions: {list(AVAILABLE_LOSS_FUNCTIONS.keys())}"
            )
        return v


def create_loss_function(config: ClassificationLossFunctionConfig) -> Module:
    """
    Create a loss function based on the configuration provided in LossFunctionConfig.

    Args:
        config: Configuration object specifying the name and arguments of the loss function.

    Returns:
        Module: A PyTorch loss function module.
    """
    loss_class = AVAILABLE_LOSS_FUNCTIONS.get(config.name)

    if loss_class is None:
        raise ValueError(
            f"Loss function '{config.name}' is not implemented.\n"
            f"Available loss functions: {list(AVAILABLE_LOSS_FUNCTIONS.keys())}"
        )

    loss_function = loss_class(**config.extra_arguments)

    return loss_function
