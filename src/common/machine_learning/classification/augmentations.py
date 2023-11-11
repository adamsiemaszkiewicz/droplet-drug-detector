# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from kornia.augmentation import (
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


class AugmentationConfig(BaseModel):
    names: List[str]
    arguments: List[Dict[str, Any]]

    @validator("arguments")
    def check_lengths(cls, v: List[Dict[str, Any]], values: Dict[str, Any]) -> None:
        """
        Validates that the 'names' and 'arguments' lists are of the same length.
        """
        if "names" in values and len(values["names"]) != len(v):
            raise ValueError("The number of augmentation names and arguments must be the same.")


INTENSITY_AUGMENTATIONS = {
    "color_jitter": ColorJitter,
    "random_brightness": RandomBrightness,
    "random_contrast": RandomContrast,
    "random_gamma": RandomGamma,
    "random_gaussian_noise": RandomGaussianNoise,
    "random_saturation": RandomSaturation,
}
GEOMETRIC_AUGMENTATIONS = {
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
ALL_AUGMENTATIONS = {**INTENSITY_AUGMENTATIONS, **GEOMETRIC_AUGMENTATIONS}


def create_augmentations(config: AugmentationConfig) -> Sequential:
    """
    Creates a sequence of transformations based on the augmentation names and individual arguments provided.
    """
    augmentations = []
    for name, args in zip(config.names, config.arguments):
        augmentation_class = ALL_AUGMENTATIONS.get(name)

        if not augmentation_class:
            raise ValueError(f"Augmentation {name} is not recognized or supported.")

        augmentation = augmentation_class(**args)
        augmentations.append(augmentation)

    augmentation_sequence = Sequential(*augmentations)

    return augmentation_sequence
