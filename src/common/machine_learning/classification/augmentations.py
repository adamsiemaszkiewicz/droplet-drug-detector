# -*- coding: utf-8 -*-
from typing import Any, Dict, List

import torch
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
from torch import Tensor
from torch.nn import Module, Sequential

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class AugmentationConfig(BaseModel):
    augmentation_names: List[str]
    augmentation_args: Dict[str, Any]

    @validator("augmentation_args", always=True)
    def check_matching_lengths(cls, v: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that the number of augmentations matches the number of provided arguments.
        """

        if "augmentations" in values and len(v) != len(values["augmentations"]):
            raise ValueError("The number of augmentations must match the number of augmentation arguments")
        return v


class Augmentation(Module):
    """
    Module for applying data augmentations to input data.

    List of all available augmentations: https://kornia.readthedocs.io/en/latest/augmentation.module.html

    Attributes:
        config (AugmentationConfig): Configuration containing augmentation specifications.
        transforms (Module): A sequence of transforms compiled from the augmentation names and arguments.
    """

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

    def __init__(self, config: AugmentationConfig) -> None:
        """
        Initializes the augmentation module with the given configuration.

        Args:
            config (AugmentationConfig): The configuration for the augmentations.
        """
        super().__init__()
        self.config = config
        self.transforms = self.create_transforms(augmentation_names=config.augmentation_names)
        _logger.info(f"Initializing augmentations: {config.augmentation_names}")

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the sequence of augmentations to the input tensor.

        Args:
            x (Tensor): The input tensor to augment.

        Returns:
            Tensor: The augmented tensor.
        """
        return self.transforms(x)  # BxCxHxW

    def create_transforms(self, augmentation_names: List[str]) -> Module:
        """
        Creates a sequence of transformations based on the provided augmentation names and arguments.

        Args:
            augmentation_names (List[str]): List of names of the augmentations to apply.

        Returns:
            Module: A torch.nn.Sequential module containing all the transformations.
        """
        augmentations = [
            self.configure_augmentations(name=name, **self.config.augmentation_args.get(name, {}))
            for name in augmentation_names
        ]
        return Sequential(*augmentations)

    def configure_augmentations(self, name: str, **kwargs: Any) -> Module:
        """
        Configures individual augmentations based on name and provided kwargs.

        Args:
            name (str): The name of the augmentation.
            **kwargs: Arbitrary keyword arguments for the augmentation's constructor.

        Returns:
            Module: The instantiated augmentation module.

        Raises:
            NotImplementedError: If the augmentation is not implemented.
        """
        if name in self.ALL_AUGMENTATIONS:
            return self.ALL_AUGMENTATIONS[name](**kwargs)
        else:
            raise NotImplementedError(f"Augmentation {name} is not implemented")
