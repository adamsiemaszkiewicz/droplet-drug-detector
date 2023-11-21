# -*- coding: utf-8 -*-
from typing import Dict, Type

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

AVAILABLE_AUGMENTATIONS: Dict[str, Type[AugmentationBase2D]] = {**INTENSITY_AUGMENTATIONS, **GEOMETRIC_AUGMENTATIONS}

REQUIRED_ARGUMENTS: Dict[str, str] = {
    "random_affine": "degrees",
    "random_crop": "size",
    "random_rotation": "degrees",
    "random_resized_crop": "size",
}
