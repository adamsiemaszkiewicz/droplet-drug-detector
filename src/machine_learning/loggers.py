# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Type

from lightning.pytorch.loggers import CSVLogger, Logger, MLFlowLogger, TensorBoardLogger
from pydantic import BaseModel, validator

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

AVAILABLE_LOGGERS: Dict[str, Type[Logger]] = {
    "csv": CSVLogger,
    "mlflow": MLFlowLogger,
    "tensorboard": TensorBoardLogger,
}


class LoggersConfig(BaseModel):
    """
    Configuration for creating a list of loggers based on their names and configurations.
    """

    name_list: List[str]
    config_list: Optional[List[Dict[str, Any]]] = None

    @validator("config_list", pre=True, always=True)
    def fill_empty_config_list(cls, v: Optional[List[Dict[str, Any]]], values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Ensures `config_list` is populated with empty dictionaries if None is provided.
        """
        name_list = values.get("name_list", [])
        return [{}] * len(name_list) if v is None else v

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Ensures each name in `name_list` corresponds to an available logger.
        """
        if v not in AVAILABLE_LOGGERS:
            raise ValueError(f"Logger '{v}' is not available. Available loggers: {list(AVAILABLE_LOGGERS.keys())}")
        return v

    @validator("config_list")
    def validate_list_lengths(cls, v: List[Dict[str, Any]], values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Checks that `config_list` has the same length as `name_list`.
        """
        name_list = values.get("name_list", [])
        if len(name_list) != len(v):
            raise ValueError("The length of 'name_list' and 'config_list' must be the same.")
        return v


def create_logger(logger_name: str, config: Dict[str, Any]) -> Logger:
    """
    Creates a logger instance from a given name and configuration.

    Args:
        logger_name (str): The name of the logger to create.
        config (Dict[str, Any]): A dictionary of configuration for the logger's constructor.

    Returns:
        Logger: An instance of the specified logger class.
    """
    logger_class = AVAILABLE_LOGGERS.get(logger_name)
    if not logger_class:
        raise ValueError(f"Logger '{logger_name}' is not defined in AVAILABLE_LOGGERS.")
    try:
        logger = logger_class(**config)
    except TypeError as e:
        raise ValueError(f"Incorrect arguments for {logger_name}: {e}")

    _logger.info(f"Created logger {logger_name} with configuration: {config}")

    return logger


def create_loggers(config: LoggersConfig) -> List[Logger]:
    """
    Creates a list of logger instances from a LoggersConfig instance.

    Args:
        config (LoggersConfig): The configuration object containing logger settings.

    Returns:
        List[LightningLoggerBase]: A list of logger instances configured as per the LoggersConfig.
    """
    _logger.info(f"Creating loggers with the following configurations: {config.name_list}")

    loggers = [
        create_logger(logger_name=name, config=cfg) for name, cfg in zip(config.name_list, config.config_list or [])
    ]

    _logger.info("Loggers successfully created.")
    return loggers


# Example usage:
# Define your logger configuration here. For example:
# my_logger_config = LoggersConfig(name_list=["tensorboard", "csv"],
#                                  config_list=[{"save_dir": "logs/", "name": "my_experiment"},
#                                               {"save_dir": "logs/", "name": "my_experiment"}])
# loggers = create_loggers(my_logger_config)
# These 'loggers' can now be used with a PyTorch Lightning Trainer instance.
