# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

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
    Configuration for creating a list of loggers.
    """

    name_list: List[str]
    save_dir: Union[str, Path]
    extra_arguments_list: List[Dict[str, Any]] = []

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Validates if all metric names are implemented.
        """
        if v not in AVAILABLE_LOGGERS:
            raise ValueError(f"Logger '{v}' is not implemented. Available loggers: {list(AVAILABLE_LOGGERS.keys())}")
        return v

    @validator("extra_arguments_list", pre=True, always=True)
    def default_extra_arguments(
        cls, v: List[Optional[Dict[str, Any]]], values: Dict[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Ensures a correct length of `extra_arguments_list` if none are provided.
        """
        if not v:
            name_list = values.get("name_list", [])
            return [{} for _ in name_list]
        return v

    @validator("extra_arguments_list")
    def validate_number_of_extra_arguments(
        cls, v: List[Dict[str, Any]], values: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Ensures that all required arguments are provided.
        """
        name_list = values.get("name_list")
        if name_list is not None and len(v) != len(name_list):
            raise ValueError(
                f"The number of extra arguments ({len(v)}) does not match the number of loggers ({len(name_list)})."
            )
        return v

    @validator("extra_arguments_list", always=True, each_item=True)
    def validate_missing_extra_arguments(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replaces missing extra arguments with empty dictionaries
        """
        if v is None:
            return {}
        return v


def create_logger(
    logger_class: Type[Logger], save_dir: Union[str, Path], arguments: Optional[Dict[str, Any]] = None
) -> Logger:
    """
    Creates a logger based on the configuration.

    Args:
        logger_class (Type[Logger]): The logger class.
        save_dir (Union[str, Path]): The directory where the logs will be saved.
        arguments (Optional[Dict[str, Any]]: Additional arguments specific to the metric.

    Returns:
        Logger: An instance of the specified logger class.
    """
    config = {"save_dir": save_dir}
    config.update(arguments or {})

    logger = logger_class(save_dir=save_dir, **arguments)

    _logger.info(f"Created logger '{logger_class.__name__}' with the following configuration: {config}")

    return logger


def create_loggers(config: LoggersConfig) -> List[Logger]:
    """
    Creates a list of logger instances from a LoggersConfig instance.

    Args:
        config (LoggersConfig): A LoggersConfig instance.

    Returns:
        List[LightningLoggerBase]: A list of logger instances.
    """
    _logger.info(f"Creating {len(config.name_list)} loggers.")

    loggers = [
        create_logger(logger_class=AVAILABLE_LOGGERS[name], save_dir=config.save_dir, arguments=args)
        for name, args in zip(config.name_list, config.extra_arguments_list)
    ]

    _logger.info("Metrics configured successfully.")

    return loggers
