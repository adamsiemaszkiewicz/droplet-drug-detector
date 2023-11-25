# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from lightning.pytorch.loggers import Logger

from src.common.utils.logger import get_logger
from src.machine_learning.loggers.config import LoggersConfig
from src.machine_learning.loggers.types import AVAILABLE_LOGGERS

_logger = get_logger(__name__)


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

    logger = logger_class(**config)

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
