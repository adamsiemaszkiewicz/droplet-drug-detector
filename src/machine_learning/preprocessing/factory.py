# -*- coding: utf-8 -*-
from typing import Any, Dict, Type

import torch
from kornia.augmentation import AugmentationBase2D
from torch import Tensor
from torch.nn import Module, Sequential

from src.common.utils.logger import get_logger
from src.machine_learning.preprocessing.config import PreprocessingConfig
from src.machine_learning.preprocessing.types import AVAILABLE_TRANSFORMATIONS

_logger = get_logger(__name__)


class DataPreprocessor(Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    def __init__(self, config: PreprocessingConfig) -> None:
        super().__init__()

        self.transforms = self.create_transform_sequence(config=config)

    def create_transform_sequence(self, config: PreprocessingConfig) -> Sequential:
        """
        Creates a sequential pipeline of transformations from a PreprocessingConfig instance.

        Args:
            config (PreprocessingConfig): The configuration object containing preprocessing settings.

        Returns:
            Sequential: A Sequential object containing the configured preprocessing pipeline.
        """
        if config.extra_arguments_list is None:
            raise ValueError("'extra_arguments_list' cannot be None")

        _logger.info(f"Creating preprocessing sequence with the following transformations: {config.name_list}")

        transformations = [
            create_transformation(transformation_class=AVAILABLE_TRANSFORMATIONS[name], arguments=arguments)
            for name, arguments in zip(config.name_list, config.extra_arguments_list)
        ]

        _logger.info("Preprocessing sequence successfully created.")
        return Sequential(*transformations)

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to apply the sequence of transformations defined in self.transforms to the input tensor x.

        Args:
        x (Tensor): The input data tensor to which the transformations will be applied.

        Returns:
        torch.Tensor: The transformed data tensor.
        """
        return self.transforms(x)


def create_transformation(
    transformation_class: Type[AugmentationBase2D], arguments: Dict[str, Any]
) -> AugmentationBase2D:
    """
    Creates a transformation from a given class with provided arguments.

    Args:
        transformation_class (Type[AugmentationBase2D]): The class of the transformation to create.
        arguments (Dict[str, Any]): A dictionary of arguments for the transformation's constructor.

    Returns:
        AugmentationBase2D: An instance of the specified transformation class.

    Raises:
        ValueError: If the provided arguments are not suitable for the transformation class.
    """
    try:
        transformation = transformation_class(**arguments)
    except TypeError as e:
        raise ValueError(f"Incorrect arguments for {transformation_class.__name__}: {e}")
    _logger.info(f"Created {transformation_class.__name__} with arguments: {arguments}")

    return transformation
