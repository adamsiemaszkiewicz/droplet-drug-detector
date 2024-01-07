# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from pydantic import BaseModel
from torch import Tensor
from torchvision.transforms import ToTensor

from src.aml.components.classificator_training.data import ClassificationDataModule, DropletDrugClassificationDataset
from src.common.utils.logger import get_logger
from src.machine_learning.preprocessing.factory import DataPreprocessor

_logger = get_logger(__name__)


class RegressionDataConfig(BaseModel):
    """
    Configuration class for regression data.

    Attrs:
        dataset_dir (Path): The directory where the dataset is located.
        val_split (float): The fraction of the dataset to use as validation set.
        test_split (float): The fraction of the dataset to use as test set.
        batch_size (int): The number of samples per batch.
    """

    dataset_dir: Path
    val_split: float
    test_split: float
    batch_size: int


class DropletDrugRegressionDataset(DropletDrugClassificationDataset):
    """A PyTorch Dataset class for loading images for dried droplet drug concentration regression."""

    def __init__(self, root_dir: Path, preprocessor: Optional[Callable] = None):
        super().__init__(root_dir, preprocessor)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get the image and its concentration at the given index.

        Args:
            idx (int): The index of the sample to return.

        Returns:
            Tuple[Tensor, Tensor]: The image and its concentration.
        """
        image_path, _, concentration = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = ToTensor()(image)

        if self.preprocessor:
            image = self.preprocessor(image).squeeze(0)

        concentration = torch.tensor(concentration, dtype=torch.float)

        return image, concentration


class RegressionDataModule(ClassificationDataModule):
    """A LightningDataModule specifically for regression tasks on DropletDrug dataset."""

    def __init__(self, config: RegressionDataConfig, preprocessor: Optional[DataPreprocessor] = None) -> None:
        super().__init__(config, preprocessor)
        self.train_dataset: Optional[DropletDrugRegressionDataset] = None
        self.val_dataset: Optional[DropletDrugRegressionDataset] = None
        self.test_dataset: Optional[DropletDrugRegressionDataset] = None
