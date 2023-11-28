# -*- coding: utf-8 -*-
from pathlib import Path

from pydantic import BaseModel

from src.common.consts.directories import DATA_DIR
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class ClassificationDataConfig(BaseModel):
    dataset_dir: Path = DATA_DIR / "dataset"
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32
