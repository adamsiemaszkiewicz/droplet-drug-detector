# -*- coding: utf-8 -*-
from typing import Dict, List, Type, Union

from kornia.augmentation import AugmentationBase2D, CenterCrop, Normalize, Resize

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_TRANSFORMATIONS: Dict[str, Type[AugmentationBase2D]] = {
    "center_crop": CenterCrop,
    "normalize": Normalize,
    "resize": Resize,
}

REQUIRED_ARGUMENTS: Dict[str, Union[str, List[str]]] = {
    "normalize": ["mean", "std"],
    "resize": "size",
}
