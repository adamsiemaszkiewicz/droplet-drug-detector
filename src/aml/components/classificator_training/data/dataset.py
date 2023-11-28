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

    Attributes:
        root_dir (Path): The dataset directory
        transform (Optional[Callable]): Optional transform to be applied on a sample.
        CLASSES (Dict[int, str]): Dictionary mapping class indices to class names.
    """

    CLASSES = {
        0: "gelatin-capsule",
        1: "lactose",
        2: "methyl-cellulose",
        3: "naproxen",
        4: "pearlitol",
        5: "polyvinyl-alcohol",
    }

    def __init__(self, root_dir: Path, transform: Optional[Callable] = None):
        """
        Args:
            root_dir (Path): Directory with all the images.
            transform (Optional[Callable]): Optional transform to be applied on a sample.

        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            sample (Tuple[Tensor, Tensor]): The image and its label.
        """
        image_path, class_id = self.samples[idx]
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
        class_indices = [class_id for _, class_id in self.samples]
        class_balance = Counter(class_indices)
        return dict(sorted(class_balance.items()))

    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        for class_idx, class_name in self.CLASSES.items():
            class_dirs = [
                path
                for path in self.root_dir.glob(f"{class_name}_*")
                if path.is_dir() and path.stem.startswith(class_name)
            ]

            for class_dir in class_dirs:
                for image_path in class_dir.glob("*.jpg"):
                    samples.append((image_path, class_idx))

        return samples
