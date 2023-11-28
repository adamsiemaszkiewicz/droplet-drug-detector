# -*- coding: utf-8 -*-
from collections import Counter
from typing import Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.aml.components.classificator_training.data.config import ClassificationDataConfig
from src.aml.components.classificator_training.data.dataset import DropletDrugClassificationDataset
from src.common.utils.logger import get_logger
from src.common.utils.os import get_cpu_worker_count
from src.machine_learning.preprocessing.factory import DataPreprocessor

_logger = get_logger(__name__)


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

    @property
    def class_balance(self) -> Dict[int, int]:
        """
        Get the overall class balance across training, validation, and test datasets.

        Returns:
            Dict[int, int]: A dictionary representing the overall class balance.
        """
        overall_balance: Counter = Counter()
        if self.train_dataset is not None:
            overall_balance.update(self.train_dataset.class_balance)
        if self.val_dataset is not None:
            overall_balance.update(self.val_dataset.class_balance)
        if self.test_dataset is not None:
            overall_balance.update(self.test_dataset.class_balance)

        return dict(overall_balance)

    @property
    def train_class_balance(self) -> Dict[int, int]:
        """
        Get the class balance of the training dataset.

        Returns:
            Dict[int, int]: A dictionary representing class balance.
        """
        if self.train_dataset is not None:
            return self.train_dataset.class_balance
        return {}

    @property
    def val_class_balance(self) -> Dict[int, int]:
        """
        Get the class balance of the validation dataset.

        Returns:
            Dict[int, int]: A dictionary representing class balance.
        """
        if self.val_dataset is not None:
            return self.val_dataset.class_balance
        return {}

    @property
    def test_class_balance(self) -> Dict[int, int]:
        """
        Get the class balance of the test dataset.

        Returns:
            Dict[int, int]: A dictionary representing class balance.
        """
        if self.test_dataset is not None:
            return self.test_dataset.class_balance
        return {}

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

        _logger.info(f"Total dataset size: {len(full_dataset)}")
        _logger.info(f"Training set size: {len(self.train_dataset)}")
        _logger.info(f"Validation set size: {len(self.val_dataset)}")
        _logger.info(f"Test set size: {len(self.test_dataset)}")

        _logger.info(f"Overall class balance: {self.class_balance}")
        _logger.info(f"Training set class balance: {self.train_class_balance}")
        _logger.info(f"Validation set class balance: {self.val_class_balance}")
        _logger.info(f"Test set class balance: {self.test_class_balance}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.cpu_workers, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.cpu_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.cpu_workers)
