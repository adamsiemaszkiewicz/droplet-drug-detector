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
    Configuration schema for loss functions with explicit common fields and provisions for additional arguments.

    Attributes:
        name: The name of the loss function.
        extra_arguments: A dictionary containing all other loss function-specific arguments.
    """

    name: str
    extra_arguments: Dict[str, Any] = {}

    @validator("extra_arguments")
    def check_extra_args(cls, v: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        if "name" in values and values["name"].startswith("asymmetric_loss") and "gamma" not in v:
            raise ValueError("gamma must be provided for asymmetric loss functions")
        return v


def create_loss_function(config: ClassificationLossFunctionConfig) -> Module:
    """
    Create a loss function based on the configuration provided in LossFunctionConfig.

    Args:
        config: Configuration object specifying the name and arguments of the loss function.

    Returns:
        Module: A PyTorch loss function module.

    Raises:
        NotImplementedError: If the loss function name is not recognized or required arguments are missing.
    """
    loss_class = AVAILABLE_LOSS_FUNCTIONS.get(config.name)
    if loss_class is None:
        valid_loss_functions = list(AVAILABLE_LOSS_FUNCTIONS.keys())
        raise ValueError(
            f"Scheduler '{config.name}' is not implemented. Available loss functions: {valid_loss_functions}"
        )

    loss_function = loss_class(**config.extra_arguments)

    return loss_function
