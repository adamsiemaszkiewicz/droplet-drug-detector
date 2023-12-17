# -*- coding: utf-8 -*-
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from lightning import LightningDataModule
from PIL import Image
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.common.consts.extensions import JPG
from src.common.utils.logger import get_logger
from src.common.utils.os import get_cpu_worker_count
from src.machine_learning.preprocessing.factory import DataPreprocessor

_logger = get_logger(__name__)


class ClassificationDataConfig(BaseModel):
    """
    Configuration class for classification data.

    Attributes:
        dataset_dir (Path): The directory where the dataset is located.
        val_split (float): The fraction of the dataset to use as validation set.
        test_split (float): The fraction of the dataset to use as test set.
        batch_size (int): The number of samples per batch.
    """

    dataset_dir: Path
    val_split: float
    test_split: float
    batch_size: int


class DropletDrugClassificationDataset(Dataset):
    """
    A PyTorch Dataset class for loading images for dried droplet drug classification.

    Attributes:
        root_dir (Path): The dataset directory
        preprocessor (Optional[Callable]): Optional transform to be applied on a sample.
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

    def __init__(self, root_dir: Path, preprocessor: Optional[Callable] = None):
        """
        Args:
            root_dir (Path): Directory with all the images.
            preprocessor (Optional[Callable]): Optional preprocessing transform to be applied on a sample.

        """
        self.root_dir = root_dir
        self.preprocessor = preprocessor
        self.samples = self._load_samples()

    @property
    def class_balance(self) -> Dict[int, int]:
        """
        The class balance of the dataset.

        Returns:
            Dict[int, int]: A dictionary mapping each class index to its count in the dataset.
        """
        class_indices = [class_id for _, class_id, _ in self.samples]
        class_balance = Counter(class_indices)
        return dict(sorted(class_balance.items()))

    @property
    def concentration_balance(self) -> Dict[float, int]:
        """
        The concentration balance of the dataset.

        Returns:
            Dict[float, int]: A dictionary mapping each concentration to its count in the dataset.
        """
        concentrations = [concentration for _, _, concentration in self.samples]
        concentration_balance = Counter(concentrations)
        return dict(sorted(concentration_balance.items()))

    def _load_samples(self) -> List[Tuple[Path, int, float]]:
        samples = []
        for class_idx, class_name in self.CLASSES.items():
            class_dirs = [
                path
                for path in self.root_dir.glob(f"{class_name}_*")
                if path.is_dir() and path.stem.startswith(class_name)
            ]

            class_samples = []
            for class_dir in class_dirs:
                concentration_string = class_dir.name.split("_")[-2]
                if concentration_string.endswith("mgml"):
                    concentration = float(concentration_string[:-4])
                else:
                    raise ValueError(f"Invalid concentration string: {concentration_string}")
                for image_path in class_dir.glob(f"*{JPG}"):
                    class_samples.append((image_path, class_idx, concentration))

            samples.extend(class_samples)

        return samples

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get the image and its label at the given index.

        Args:
            idx (int): The index of the sample to return.

        Returns:
            Tuple[Tensor, Tensor]: The image and its label.
        """
        image_path, class_id, _ = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = ToTensor()(image)

        if self.preprocessor:
            image = self.preprocessor(image).squeeze(0)

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

    @property
    def concentration_balance(self) -> Dict[float, int]:
        """
        Get the overall concentration balance across training, validation, and test datasets.

        Returns:
            Dict[float, int]: A dictionary representing the overall concentration balance.
        """
        overall_balance: Counter = Counter()
        if self.train_dataset is not None:
            overall_balance.update(self.train_dataset.concentration_balance)
        if self.val_dataset is not None:
            overall_balance.update(self.val_dataset.concentration_balance)
        if self.test_dataset is not None:
            overall_balance.update(self.test_dataset.concentration_balance)

        return dict(overall_balance)

    @property
    def train_concentration_balance(self) -> Dict[float, int]:
        """
        Get the concentration balance of the training dataset.

        Returns:
            Dict[float, int]: A dictionary representing concentration balance.
        """
        if self.train_dataset is not None:
            return self.train_dataset.concentration_balance
        return {}

    @property
    def val_concentration_balance(self) -> Dict[float, int]:
        """
        Get the concentration balance of the validation dataset.

        Returns:
            Dict[float, int]: A dictionary representing concentration balance.
        """
        if self.val_dataset is not None:
            return self.val_dataset.concentration_balance
        return {}

    @property
    def test_concentration_balance(self) -> Dict[float, int]:
        """
        Get the concentration balance of the test dataset.

        Returns:
            Dict[float, int]: A dictionary representing concentration balance.
        """
        if self.test_dataset is not None:
            return self.test_dataset.concentration_balance
        return {}

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the data module.

        This method is responsible for preparing the datasets for training, validation, and testing.
        It performs stratified split on the full dataset to create these subsets and checks for data leaks between them.

        Args:
            stage (Optional[str]): The stage for which setup is being performed.
                                   It can be either 'fit' or 'test'. If None, both 'fit' and 'test' are considered.
        """
        # Create a full dataset without transforms to split it first
        full_dataset = DropletDrugClassificationDataset(root_dir=self.dataset_dir)

        # Perform stratified split
        train_indices, val_indices, test_indices = self.stratified_split(full_dataset)

        # Assign the subset indices to the respective datasets
        self.train_dataset = DropletDrugClassificationDataset(root_dir=self.dataset_dir, preprocessor=self.preprocessor)
        self.val_dataset = DropletDrugClassificationDataset(root_dir=self.dataset_dir, preprocessor=self.preprocessor)
        self.test_dataset = DropletDrugClassificationDataset(root_dir=self.dataset_dir, preprocessor=self.preprocessor)

        self.train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
        self.val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
        self.test_dataset.samples = [full_dataset.samples[i] for i in test_indices]

        self.check_data_leak(self.train_dataset, self.val_dataset, self.test_dataset)

        self.log_dataset_details()

    def stratified_split(self, dataset: DropletDrugClassificationDataset) -> Tuple[List[int], List[int], List[int]]:
        """
        Perform a stratified split on the dataset.

        Args:
            dataset (DropletDrugClassificationDataset): The dataset to split.

        Returns:
            Tuple[List[int], List[int], List[int]]: Indices for the train, validation, and test sets.
        """

        # Extract labels and concentrations
        labels, concentrations = zip(*[(sample[1], sample[2]) for sample in dataset.samples])

        # Convert to a multi-label format
        multi_labels = np.array([[label, concentration] for label, concentration in zip(labels, concentrations)])

        # Create Multilabel Stratified Splitter for train and test_val split
        splitter_test_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.val_split + self.test_split)
        train_index, test_val_index = next(splitter_test_val.split(np.zeros(len(multi_labels)), multi_labels))

        # Further split for validation and test sets
        test_split_adjusted = self.test_split / (self.val_split + self.test_split)
        splitter_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_split_adjusted)
        val_index, test_index = next(splitter_test.split(test_val_index, multi_labels[test_val_index]))

        return train_index, test_val_index[val_index], test_val_index[test_index]

    @staticmethod
    def check_data_leak(
        train_dataset: DropletDrugClassificationDataset,
        val_dataset: DropletDrugClassificationDataset,
        test_dataset: DropletDrugClassificationDataset,
    ) -> None:
        """
        Check for data leaks between datasets.

        Args:
            train_dataset (DropletDrugClassificationDataset): Training dataset.
            val_dataset (DropletDrugClassificationDataset): Validation dataset.
            test_dataset (DropletDrugClassificationDataset): Test dataset.
        """
        train_paths = {sample[0] for sample in train_dataset.samples}
        val_paths = {sample[0] for sample in val_dataset.samples}
        test_paths = {sample[0] for sample in test_dataset.samples}

        # Check for overlaps
        if (
            train_paths.intersection(val_paths)
            or train_paths.intersection(test_paths)
            or val_paths.intersection(test_paths)
        ):
            raise ValueError("Data leak detected between subsets!")

    def log_dataset_details(self) -> None:
        """
        Log dataset details.
        """
        _logger.info(
            f"Total dataset size: "
            f"{sum(len(ds) for ds in [self.train_dataset, self.val_dataset, self.test_dataset] if ds is not None)}"
        )
        _logger.info(f"Training set size: {len(self.train_dataset) if self.train_dataset is not None else 0}")
        _logger.info(f"Validation set size: {len(self.val_dataset) if self.val_dataset is not None else 0}")
        _logger.info(f"Test set size: {len(self.test_dataset) if self.test_dataset is not None else 0}")

        _logger.info(f"Overall class balance: {self.class_balance}")
        _logger.info(f"Training set class balance: {self.train_class_balance}")
        _logger.info(f"Validation set class balance: {self.val_class_balance}")
        _logger.info(f"Test set class balance: {self.test_class_balance}")

        _logger.info(f"Overall concentration balance: {self.concentration_balance}")
        _logger.info(f"Training set concentration balance: {self.train_concentration_balance}")
        _logger.info(f"Validation set concentration balance: {self.val_concentration_balance}")
        _logger.info(f"Test set concentration balance: {self.test_concentration_balance}")

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.cpu_workers, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the validation dataset.
        """
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.cpu_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test dataset.
        """
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.cpu_workers)

    def calculate_dataset_stats(self, dataset: Dataset) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        Calculate the mean and standard deviation of the dataset.

        Args:
            dataset (Dataset): The dataset for which to calculate statistics.

        Returns:
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
            A tuple containing the mean & the standard deviation for each of the RGB channels.
        """

        data_loader = DataLoader(dataset, num_workers=self.cpu_workers)

        sum_pixels = np.zeros(3)
        sum_sq_pixels = np.zeros(3)
        total_pixels = 0

        for batch in tqdm(data_loader, desc="Calculating dataset statistics"):
            images, _ = batch
            images = images.view(images.size(0), 3, -1)
            sum_batch = images.sum(axis=[0, 2]).numpy()
            sum_pixels += sum_batch
            sum_sq_pixels += (images**2).sum(axis=[0, 2]).numpy()
            total_pixels += images.size(0) * images.size(2)

        mean = tuple(float(x) for x in sum_pixels / total_pixels)
        std = tuple(float(x) for x in np.sqrt(sum_sq_pixels / total_pixels - np.array(mean) ** 2))

        _logger.info(f"Dataset mean: {mean}\nDataset standard deviation: {std}")

        return mean, std
