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
from pydantic import validator
from torch.nn import Sequential

from src.common.utils.logger import get_logger
from src.configs.base import BaseAugmentationsConfig

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


class AugmentationsConfig(BaseAugmentationsConfig):
    """
    Configuration for creating a sequence of augmentations.
    """

    name_list: List[str]
    extra_arguments_list: Optional[List[Dict[str, Any]]] = None

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Validates if all augmentation names are implemented.
        """
        if v not in AVAILABLE_AUGMENTATIONS:
            raise ValueError(
                f"Augmentation '{v}' is not implemented.\n"
                f"Available augmentations: {list(AVAILABLE_AUGMENTATIONS.keys())}"
            )
        return v

    @validator("extra_arguments_list", pre=True, always=True)
    def default_extra_arguments(
        cls, v: List[Optional[Dict[str, Any]]], values: Dict[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Ensures a correct length of `extra_arguments_list` if none are provided.
        """
        if not v:
            name_list = values.get("name_list", [])
            return [{} for _ in name_list]
        return v

    @validator("extra_arguments_list")
    def validate_number_of_extra_arguments(
        cls, v: List[Dict[str, Any]], values: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Ensures that all required arguments are provided.
        """
        name_list = values.get("name_list")
        if name_list is not None and len(v) != len(name_list):
            raise ValueError(
                f"The number of extra arguments ({len(v)}) does not match the number of loggers ({len(name_list)})."
            )
        return v

    @validator("extra_arguments_list", always=True, each_item=True)
    def validate_missing_extra_arguments(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replaces missing extra arguments with empty dictionaries
        """
        if v is None:
            return {}
        return v

    @validator("extra_arguments_list", always=True)
    def validate_required_augmentation_arguments(
        cls, extra_args_list: List[Optional[Dict[str, Any]]], values: Dict[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Validates if the required arguments for each chosen augmentation are provided.
        """
        name_list = values.get("name_list", [])
        if not extra_args_list:
            extra_args_list = [{} for _ in name_list]

        for name, extra_args in zip(name_list, extra_args_list):
            if name in REQUIRED_ARGUMENTS:
                required_arg = REQUIRED_ARGUMENTS[name]
                if extra_args is None or required_arg not in extra_args:
                    raise ValueError(f"Required argument '{required_arg}' for augmentation '{name}' is missing.")

        return extra_args_list


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
