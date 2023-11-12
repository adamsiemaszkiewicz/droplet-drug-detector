# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Type

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
    """

    name_list: List[str]
    extra_arguments_list: Optional[List[Dict[str, Any]]] = None

    @validator("extra_arguments_list", pre=True)
    def fill_empty_extra_arguments_list(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pre-validator to ensure `extra_arguments_list` is populated with empty dictionaries if None.
        """
        name_list, extra_arguments_list = values.get("name_list"), values.get("extra_arguments_list")
        if extra_arguments_list is None and name_list is not None:
            extra_arguments_list = [{} for _ in name_list]
        values["extra_arguments_list"] = extra_arguments_list
        return values

    @validator("name_list", "extra_arguments_list")
    def validate_list_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validator to check the length of `name_list` and `extra_arguments_list` are the same.
        """
        name_list, extra_arguments_list = values.get("name_list"), values.get("extra_arguments_list")
        if name_list is not None and extra_arguments_list is not None:
            if len(name_list) != len(extra_arguments_list):
                raise ValueError("The length of 'name_list' and 'extra_arguments_list' must be the same.")
        return values

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Validator to ensure each name in `name_list` corresponds to a valid augmentation.
        """
        if v not in ALL_AUGMENTATIONS:
            raise ValueError(
                f"Augmentation '{v}' is not implemented. Available augmentations: {list(ALL_AUGMENTATIONS.keys())}"
            )
        return v


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
        Sequential: A torch.nn.Sequential object containing the configured augmentation pipeline.

    Raises:
        ValueError: If `extra_arguments_list` is None, which is required to be a list for the pipeline.
    """
    if config.extra_arguments_list is None:
        raise ValueError("'extra_arguments_list' cannot be None")

    augmentations = [
        create_augmentation(augmentation_class=ALL_AUGMENTATIONS[name], arguments=arguments)
        for name, arguments in zip(config.name_list, config.extra_arguments_list)
    ]
    _logger.info(f"Augmentation sequence created with {len(augmentations)} augmentations.")
    return Sequential(*augmentations)
