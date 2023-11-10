# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, List, Type, Union


class FieldValidators:
    """
    A utility class for converting strings to various data types and checking values.
    """

    @staticmethod
    def convert_str_to_int(value: str) -> int:
        """Convert a string to an integer."""
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(f"Error converting '{value}' to int: {e}")

    @staticmethod
    def convert_str_to_float(value: str) -> float:
        """Convert a string to a float."""
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Error converting '{value}' to float: {e}")

    @staticmethod
    def check_if_positive(value: Any) -> Any:
        """Check if the provided value is positive."""
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError("The value must be a positive number.")
        return value

    @staticmethod
    def convert_str_to_path(value: Union[str, Path]) -> Path:
        """Convert a string to a Path object."""
        if isinstance(value, str):
            return Path(value)
        return value

    @staticmethod
    def convert_str_to_bool(value: Union[bool, str]) -> bool:
        """Convert a string to a boolean."""
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in {"true", "1", "t", "y", "yes"}:
            return True
        elif value in {"false", "0", "f", "n", "no"}:
            return False
        else:
            raise ValueError("Value must be a boolean value")

    @staticmethod
    def convert_comma_separated_str_to_list(value: str, convert_to_type: Type) -> List:
        """Convert a comma-separated string to a list of a specified type."""
        try:
            return [convert_to_type(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as e:
            raise ValueError(f"Error converting values to {convert_to_type.__name__}: {e}")
