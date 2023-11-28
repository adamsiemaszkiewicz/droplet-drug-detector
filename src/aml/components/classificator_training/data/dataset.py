# -*- coding: utf-8 -*-
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class DropletDrugClassificationDataset(Dataset):
    """
    A PyTorch Dataset class for loading images for dried droplet drug classification.
    """

    CLASSES = {
        0: "gelatin-capsule",
        1: "lactose",
        2: "methyl-cellulose",
        3: "naproxen",
        4: "pearlitol",
        5: "polyvinyl-alcohol",
    }

    def __init__(self, image_paths: List[Path], labels: List[int], transform: Optional[Callable] = None):
        """
        Args:
            image_paths (List[Path]): List of paths to the images.
            labels (List[int]): List of class indices for each image.
            transform (Optional[Callable]): Optional transform to be applied on a sample.

        Attrs:
            transform (Optional[Callable]): Optional transform to be applied on a sample.
            image_paths (List[Path]): List of paths to the images.
            labels (List[int]): List of class indices for each image.
            CLASSES (Dict[int, str]): Dictionary mapping class indices to class names.
        """
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        assert len(self.image_paths) == len(self.labels)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            sample (Tuple[Tensor, Tensor]): The image and its label.
        """
        image_path = self.image_paths[idx]
        class_id = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        image = ToTensor()(image)

        if self.transform:
            image = self.transform(image).squeeze(0)

        label = torch.tensor(class_id, dtype=torch.long)

        return image, label

    @property
    def class_balance(self) -> Dict[int, int]:
        """
        The class balance of the dataset.

        Returns:
            Dict[int, int]: A dictionary mapping each class index to its count in the dataset.
        """
        class_balance = Counter(self.labels)
        return dict(sorted(class_balance.items()))
