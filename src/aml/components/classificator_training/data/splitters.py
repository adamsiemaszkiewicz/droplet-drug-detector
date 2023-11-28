# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Tuple

from torch import Generator
from torch.utils.data import Dataset, random_split

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split the dataset into training, validation, and test subsets.

        Args:
            dataset (Dataset): The dataset to be split.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: The training, validation, and test subsets.
        """
        pass


class RandomSplitter(BaseSplitter):
    def __init__(self, val_split: float, test_split: float):
        self.val_split = val_split
        self.test_split = test_split

    def split(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        dataset_size = len(dataset)
        val_size = int(self.val_split * dataset_size)
        test_size = int(self.test_split * dataset_size)
        train_size = dataset_size - val_size - test_size

        train_subset, val_subset, test_subset = random_split(
            dataset,
            lengths=[train_size, val_size, test_size],
            generator=Generator(),
        )
        return train_subset, val_subset, test_subset
