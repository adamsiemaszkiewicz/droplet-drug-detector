# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Type

from kornia.augmentation import (
    AugmentationBase2D,
    ColorJitter,
    RandomAffine,
    RandomBrightness,
    RandomContrast,
    RandomCrop,
    RandomElasticTransform,
    RandomErasing,
    RandomGamma,
    RandomGaussianNoise,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomResizedCrop,
    RandomRotation,
    RandomSaturation,
    RandomVerticalFlip,
)
from pydantic import BaseModel, validator
from torch.nn import Sequential

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

INTENSITY_AUGMENTATIONS: Dict[str, Type[AugmentationBase2D]] = {
    "color_jitter": ColorJitter,
    "random_brightness": RandomBrightness,
    "random_contrast": RandomContrast,
    "random_gamma": RandomGamma,
    "random_gaussian_noise": RandomGaussianNoise,
    "random_saturation": RandomSaturation,
}

GEOMETRIC_AUGMENTATIONS: Dict[str, Type[AugmentationBase2D]] = {
    "random_affine": RandomAffine,
    "random_crop": RandomCrop,
    "random_elastic_transform": RandomElasticTransform,
    "random_erasing": RandomErasing,
    "random_horizontal_flip": RandomHorizontalFlip,
    "random_vertical_flip": RandomVerticalFlip,
    "random_perspective": RandomPerspective,
    "random_rotation": RandomRotation,
    "random_resized_crop": RandomResizedCrop,
}

ALL_AUGMENTATIONS: Dict[str, Type[AugmentationBase2D]] = {**INTENSITY_AUGMENTATIONS, **GEOMETRIC_AUGMENTATIONS}


class AugmentationsConfig(BaseModel):
    """
    Configuration for creating a sequence of augmentations.

    Attributes:
        names: A list of augmentation names.
        arguments: A list of dictionaries containing the arguments for each augmentation.
    """

    names: List[str]
    arguments: List[Dict[str, Any]]

    @validator("names")
    def validate_names(cls, names: List[str]) -> List[str]:
        """
        Validates that the provided augmentation names are implemented.
        """
        for name in names:
            if name not in ALL_AUGMENTATIONS:
                raise ValueError(
                    f"Augmentation '{name}' is not available. Available augmentations: {list(ALL_AUGMENTATIONS.keys())}"
                )
        return names

    @validator("arguments")
    def check_lengths(cls, v: List[Dict[str, Any]], values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validates that the 'names' and 'arguments' lists are of the same length.
        """
        if "names" in values and len(values["names"]) != len(v):
            raise ValueError("The number of augmentation names and arguments must be the same.")
        return v


def create_augmentation(name: str, args: Dict[str, Any]) -> AugmentationBase2D:
    """
    Creates a single augmentation based on the provided name and arguments.

    Args:
        name: The name of the augmentation to create.
        args: A dictionary of arguments for the augmentation's constructor.

    Returns:
        An instance of the requested augmentation class initialized with the provided arguments.

    Raises:
        ValueError: If the augmentation name is not recognized or the arguments are invalid.
    """
    augmentation_class = ALL_AUGMENTATIONS.get(name)
    if augmentation_class is None:
        available_augmentations = list(ALL_AUGMENTATIONS.keys())
        error_message = (
            f"Augmentation '{name}' is not implemented. Available augmentations are: {available_augmentations}"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    try:
        augmentation = augmentation_class(**args)
        _logger.info(f"Created augmentation '{name}' with the following arguments: {args}")
    except TypeError as e:
        error_message = f"Incorrect arguments for augmentation '{name}'. " f"Error: {e}"
        _logger.error(error_message)
        raise ValueError(error_message)

    return augmentation


def create_augmentations(config: AugmentationsConfig) -> Sequential:
    """
    Creates a sequence of transformations based on the augmentation names and individual arguments provided.

    Args:
        config: An instance of AugmentationConfig containing the augmentation sequence configuration.

    Returns:
        A Sequential object containing the configured augmentation sequence.
    """
    _logger.info(f"Creating augmentation sequence for the following augmentations: {config.names}")
    augmentations = [create_augmentation(name, args) for name, args in zip(config.names, config.arguments)]

    augmentation_sequence = Sequential(*augmentations)
    _logger.info("Augmentation sequence created successfully.")
    return augmentation_sequence
