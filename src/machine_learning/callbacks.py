# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Type

from lightning.pytorch.callbacks import Callback
from pydantic import BaseModel, validator

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)

# List of available callbacks for easy reference and validation
AVAILABLE_CALLBACKS: Dict[str, Type[Callback]] = {
    # "callback_name": CallbackClass,
    # Populate this dictionary with your actual callbacks, for example:
    # "early_stopping": EarlyStoppingCallback,
    # "model_checkpoint": ModelCheckpointCallback,
    # ... Add more as you define them or import them from PyTorch Lightning
}


class CallbacksConfig(BaseModel):
    """
    Configuration for creating a list of callbacks based on their names and configurations.
    """

    name_list: List[str]
    config_list: Optional[List[Dict[str, Any]]] = None

    @validator("config_list", pre=True, always=True)
    def fill_empty_config_list(cls, v: Optional[List[Dict[str, Any]]], values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Pre-validator to ensure `config_list` is populated with empty dictionaries if None.
        """
        name_list = values.get("name_list", [])
        return [{}] * len(name_list) if v is None else v

    @validator("name_list", each_item=True)
    def validate_names(cls, v: str) -> str:
        """
        Validator to ensure each name in `name_list` corresponds to a valid callback.
        """
        if v not in AVAILABLE_CALLBACKS:
            raise ValueError(
                f"Callback '{v}' is not available. Available callbacks: {list(AVAILABLE_CALLBACKS.keys())}"
            )
        return v

    @validator("config_list")
    def validate_list_lengths(cls, v: List[Dict[str, Any]], values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validator to check that `config_list` has the same length as `name_list`.
        """
        name_list = values.get("name_list", [])
        if len(name_list) != len(v):
            raise ValueError("The length of 'name_list' and 'config_list' must be the same.")
        return v


def create_callback(callback_name: str, config: Dict[str, Any]) -> Callback:
    """
    Creates a callback instance from a given name and configuration.

    Args:
        callback_name (str): The name of the callback to create.
        config (Dict[str, Any]): A dictionary of configuration for the callback's constructor.

    Returns:
        Callback: An instance of the specified callback class.

    Raises:
        ValueError: If the provided configuration is not suitable for the callback class.
    """
    callback_class = AVAILABLE_CALLBACKS.get(callback_name)
    if not callback_class:
        raise ValueError(f"Callback '{callback_name}' is not defined in AVAILABLE_CALLBACKS.")
    try:
        callback = callback_class(**config)
    except TypeError as e:
        raise ValueError(f"Incorrect arguments for {callback_name}: {e}")

    _logger.info(f"Created callback {callback_name} with configuration: {config}")
    return callback


def create_callbacks(config: CallbacksConfig) -> List[Callback]:
    """
    Creates a list of callback instances from a CallbacksConfig instance.

    Args:
        config (CallbacksConfig): The configuration object containing callback settings.

    Returns:
        List[Callback]: A list of callback instances configured as per the CallbacksConfig.
    """
    _logger.info(f"Creating callbacks with the following configurations: {config.name_list}")

    callbacks = [
        create_callback(callback_name=name, config=cfg) for name, cfg in zip(config.name_list, config.config_list or [])
    ]

    _logger.info("Callbacks successfully created.")
    return callbacks
