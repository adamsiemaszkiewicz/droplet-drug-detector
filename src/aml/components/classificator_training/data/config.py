# -*- coding: utf-8 -*-
from pathlib import Path

from pydantic import BaseModel

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class ClassificationDataConfig(BaseModel):
    dataset_dir: Path
    val_split: float
    test_split: float
    batch_size: int
