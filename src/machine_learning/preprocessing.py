# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Type, Union

import torch
from kornia.augmentation import AugmentationBase2D, CenterCrop, Normalize, Resize
from kornia.color import rgb_to_grayscale
from pydantic import validator
from torch import Tensor
from torch.nn import Module, Sequential

from src.common.utils.logger import get_logger
from src.configs.base import BasePreprocessingConfig

_logger = get_logger(__name__)

AVAILABLE_TRANSFORMATIONS: Dict[str, Union[Type[AugmentationBase2D], Type[rgb_to_grayscale]]] = {
    "center_crop": CenterCrop,
    "normalize": Normalize,
    "resize": Resize,
    "rgb_to_grayscale": rgb_to_grayscale,
}

REQUIRED_ARGUMENTS: Dict[str, Union[str, List[str]]] = {
    "normalize": ["mean", "std"],
    "resize": "size",
}


class PreprocessingConfig(BasePreprocessingConfig):
    """
    Configuration for creating a sequence of preprocessing transformations.
    """

    name_list: List[str]
    extra_arguments_list: Optional[List[Dict[str, Any]]] = None

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Validates if all transformations names are implemented.
        """
        if v not in AVAILABLE_TRANSFORMATIONS:
            raise ValueError(
                f"Transformation '{v}' is not implemented.\n"
                f"Available transformations: {list(AVAILABLE_TRANSFORMATIONS.keys())}"
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
    def validate_required_transformation_arguments(
        cls, extra_args_list: List[Optional[Dict[str, Any]]], values: Dict[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Validates if the required arguments for each chosen transformation are provided.
        """
        name_list = values.get("name_list", [])
        if not extra_args_list:
            extra_args_list = [{} for _ in name_list]

        for name, extra_args in zip(name_list, extra_args_list):
            if name in REQUIRED_ARGUMENTS:
                required_arg = REQUIRED_ARGUMENTS[name]
                if extra_args is None or required_arg not in extra_args:
                    raise ValueError(f"Required argument '{required_arg}' for transformation '{name}' is missing.")

        return extra_args_list


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
