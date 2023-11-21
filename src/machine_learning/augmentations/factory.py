# -*- coding: utf-8 -*-
from typing import Any, Dict, Type

from kornia.augmentation import AugmentationBase2D
from torch.nn import Sequential

from src.common.utils.logger import get_logger
from src.machine_learning.augmentations.config import AugmentationsConfig
from src.machine_learning.augmentations.types import AVAILABLE_AUGMENTATIONS

_logger = get_logger(__name__)


def create_augmentation(augmentation_class: Type[AugmentationBase2D], arguments: Dict[str, Any]) -> AugmentationBase2D:
    """
    Creates an augmentation from a given class with provided arguments.

    Args:
        augmentation_class (Type[AugmentationBase2D]): The class of the augmentation to create.
        arguments (Dict[str, Any]): A dictionary of arguments for the augmentation's constructor.

    Returns:
        AugmentationBase2D: An instance of the specified augmentation class.

    Raises:
        ValueError: If the provided arguments are not suitable for the augmentation class.
    """
    try:
        augmentation = augmentation_class(**arguments)
    except TypeError as e:
        raise ValueError(f"Incorrect arguments for {augmentation_class.__name__}: {e}")
    _logger.info(f"Created {augmentation_class.__name__} with arguments: {arguments}")
    return augmentation


def create_augmentations(config: AugmentationsConfig) -> Sequential:
    """
    Creates a sequential pipeline of augmentations from an AugmentationsConfig instance.

    Args:
        config (AugmentationsConfig): The configuration object containing augmentation settings.

    Returns:
        Sequential: A Sequential object containing the configured augmentation pipeline.

    Raises:
        ValueError: If `extra_arguments_list` is None, which is required to be a list for the pipeline.
    """
    if config.extra_arguments_list is None:
        raise ValueError("'extra_arguments_list' cannot be None")

    _logger.info(f"Creating augmentation sequence with the following transformations: {config.name_list}")

    augmentations = [
        create_augmentation(augmentation_class=AVAILABLE_AUGMENTATIONS[name], arguments=arguments)
        for name, arguments in zip(config.name_list, config.extra_arguments_list)
    ]
    _logger.info("Augmentation sequence successfully created.")
    return Sequential(*augmentations)
