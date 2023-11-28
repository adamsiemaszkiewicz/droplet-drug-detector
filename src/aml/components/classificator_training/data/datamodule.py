# -*- coding: utf-8 -*-
import re
from collections import Counter
from typing import Dict, Optional

from lightning import LightningDataModule
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.aml.components.classificator_training.data.config import ClassificationDataConfig
from src.aml.components.classificator_training.data.dataset import DropletDrugClassificationDataset
from src.aml.components.classificator_training.data.splitters import BaseSplitter
from src.common.consts.extensions import JPG
from src.common.utils.logger import get_logger
from src.common.utils.os import get_cpu_worker_count
from src.machine_learning.preprocessing.factory import DataPreprocessor

_logger = get_logger(__name__)


class ClassificationDataModule(LightningDataModule):
    def __init__(
        self, config: ClassificationDataConfig, splitter: BaseSplitter, preprocessor: Optional[DataPreprocessor] = None
    ) -> None:
        super().__init__()
        self.config = config
        self.splitter = splitter
        self.preprocessor = preprocessor
        self.dataset_dir = self.config.dataset_dir
        self.batch_size = self.config.batch_size
        self.cpu_workers = get_cpu_worker_count()

        self.train_dataset: Optional[DropletDrugClassificationDataset] = None
        self.val_dataset: Optional[DropletDrugClassificationDataset] = None
        self.test_dataset: Optional[DropletDrugClassificationDataset] = None

    def _load_full_dataset_dataframe(self) -> DataFrame:
        def validate_substance_name(name: str) -> str:
            if name not in DropletDrugClassificationDataset.CLASSES.values():
                raise ValueError(f"Unknown substance name: {name}")
            return name

        def validate_concentration(concentration: str) -> float:
            if not concentration.endswith("mgml"):
                raise ValueError(f"Unknown substance concentration: {concentration}")

            value_str = re.sub(pattern=r"mgml$", repl="", string=concentration)
            return float(value_str)

        def validate_lens_zoom(zoom: str) -> float:
            if not zoom.startswith("x"):
                raise ValueError(f"Unknown lens zoom: {zoom}")
            return float(zoom.lstrip("x"))

        sample_subset_dirs = sorted([path for path in self.dataset_dir.iterdir() if path.is_dir()])

        samples = []
        for subset_dir in sample_subset_dirs:
            name_split = subset_dir.name.split("_")
            substance_name = validate_substance_name(name_split[0])
            substance_idx = list(DropletDrugClassificationDataset.CLASSES.values()).index(substance_name)
            concentration = validate_concentration(name_split[1])
            lens_zoom = validate_lens_zoom(name_split[2])

            for image_path in subset_dir.glob(f"*{JPG}"):
                samples.append([image_path, substance_name, substance_idx, concentration, lens_zoom])

        return DataFrame(
            samples, columns=["image_path", "substance_name", "substance_idx", "concentration", "lens_zoom"]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        full_dataset_df = self._load_full_dataset_dataframe()

        filtered_dataset_df = full_dataset_df.copy()[full_dataset_df["concentration"] == 1.0]

        train_df, temp_df = train_test_split(
            filtered_dataset_df, test_size=self.config.val_split + self.config.test_split
        )
        val_df, test_df = train_test_split(temp_df, test_size=self.config.test_split)

        self.train_dataset = DropletDrugClassificationDataset(
            image_paths=train_df["image_path"].tolist(),
            labels=train_df["substance_idx"].tolist(),
            transform=self.preprocessor,
        )
        self.val_dataset = DropletDrugClassificationDataset(
            image_paths=val_df["image_path"].tolist(),
            labels=val_df["substance_idx"].tolist(),
            transform=self.preprocessor,
        )
        self.test_dataset = DropletDrugClassificationDataset(
            image_paths=test_df["image_path"].tolist(),
            labels=test_df["substance_idx"].tolist(),
            transform=self.preprocessor,
        )

        _logger.info(f"Total dataset size: {len(filtered_dataset_df)}")
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
