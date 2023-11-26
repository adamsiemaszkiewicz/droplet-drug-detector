# -*- coding: utf-8 -*-
from pathlib import Path

from src.common.consts.directories import CONFIGS_DIR
from src.common.consts.extensions import YAML

PROJECT_NAME: str = "droplet-drug-detector"
PROJECT_NAME_CLASSIFICATOR: str = "droplet-drug-classificator"
PROJECT_NAME_REGRESSOR: str = "droplet-drug-regressor"

MODEL_CHECKPOINTS_FOLDER_NAME: str = "checkpoints"
LOGS_FOLDER_NAME: str = "logs"

DEFAULT_CONFIG_FILE_CLASSIFICATOR: Path = CONFIGS_DIR / f"classificator-default{YAML}"
DEFAULT_CONFIG_FILE_REGRESSOR: Path = CONFIGS_DIR / f"regressor-default{YAML}"
