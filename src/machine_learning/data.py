# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor

from src.common.consts.directories import DATA_DIR
from src.common.utils.logger import get_logger
from src.common.utils.os import get_cpu_worker_count
from src.configs.base import BaseDataConfig
from src.machine_learning.preprocessing.factory import DataPreprocessor

_logger = get_logger(__name__)


class ClassificationDataConfig(BaseDataConfig):
    dataset_dir: Path = DATA_DIR / "dataset"
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32


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

        _logger.info(f"Found a total of {len(samples)} samples.")

        return samples

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


class ClassificationDataModule(LightningDataModule):
    def __init__(self, config: ClassificationDataConfig, preprocessor: Optional[DataPreprocessor] = None) -> None:
        super().__init__()
        self.config = config
        self.preprocessor = preprocessor
        self.dataset_dir = self.config.dataset_dir
        self.val_split = self.config.val_split
        self.test_split = self.config.test_split
        self.batch_size = self.config.batch_size
        self.cpu_workers = get_cpu_worker_count()

        self.train_dataset: Optional[DropletDrugClassificationDataset] = None
        self.val_dataset: Optional[DropletDrugClassificationDataset] = None
        self.test_dataset: Optional[DropletDrugClassificationDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Create a full dataset without transforms to split it first
        full_dataset = DropletDrugClassificationDataset(root_dir=self.dataset_dir)

        dataset_size = len(full_dataset)
        val_size = int(self.val_split * dataset_size)
        test_size = int(self.test_split * dataset_size)
        train_size = dataset_size - val_size - test_size

        # Create subsets
        train_subset, val_subset, test_subset = random_split(
            full_dataset,
            lengths=[train_size, val_size, test_size],
            generator=torch.Generator(),
        )

        # Now, wrap these Subsets into new Dataset instances applying the transforms
        self.train_dataset = DropletDrugClassificationDataset(root_dir=self.dataset_dir, transform=self.preprocessor)
        self.val_dataset = DropletDrugClassificationDataset(root_dir=self.dataset_dir, transform=self.preprocessor)
        self.test_dataset = DropletDrugClassificationDataset(root_dir=self.dataset_dir, transform=self.preprocessor)

        # Assign the subset indices to the respective datasets
        self.train_dataset.samples = [full_dataset.samples[i] for i in train_subset.indices]
        self.val_dataset.samples = [full_dataset.samples[i] for i in val_subset.indices]
        self.test_dataset.samples = [full_dataset.samples[i] for i in test_subset.indices]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.cpu_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.cpu_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.cpu_workers)
