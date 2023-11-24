# -*- coding: utf-8 -*-

import ast
from pathlib import Path
from typing import Dict


def str_to_path(path_string: str) -> Path:
    """
    Converts a string to a pathlib.Path object.

    Args:
        path_string (str): The string representation of the path.

    Returns:
        Path: A pathlib.Path object representing the given path.

    Raises:
        ValueError: If the string is not a valid path.
    """
    try:
        return Path(path_string)
    except ValueError:
        raise ValueError("Invalid path string")


def str_to_dict(dict_string: str) -> Dict:
    """
    Converts a string representation of a dictionary into a dictionary object.

    Args:
        dict_string (str): The string representation of the dictionary.

    Returns:
        Dict: A dictionary object parsed from the string.

    Raises:
        ValueError: If the string is not a valid dictionary representation.
    """
    try:
        return ast.literal_eval(dict_string)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid dictionary string")


def str_to_list(list_string: str) -> list:
    """
    Converts a string representation of a list into a list object.

    The string should be in a format interpretable as a Python list literal,
    e.g., "[1, 2, 3]" or "['a', 'b', 'c']".

    Args:
        list_string (str): The string representation of the list.

    Returns:
        list: A list object parsed from the string.

    Raises:
        ValueError: If the string is not a valid list representation.
    """
    try:
        return ast.literal_eval(list_string)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid list string")


def str_to_bool(bool_string: str) -> bool:
    """
    Converts a string to a boolean, accepting various representations.

    Args:
        bool_string (str): The string to convert, expected to be a boolean-like string
                           (e.g., 'True', 'False', 'Yes', 'No', '1', '0').

    Returns:
        bool: The boolean value of the string.

    Raises:
        ValueError: If the string is not a recognizable boolean representation.
    """
    true_values = {"true", "yes", "1"}
    false_values = {"false", "no", "0"}

    bool_string_lower = bool_string.lower()
    if bool_string_lower in true_values:
        return True
    elif bool_string_lower in false_values:
        return False
    else:
        raise ValueError(f"Invalid boolean string: {bool_string}")


def str_to_int(int_string: str) -> int:
    """
    Converts a string to an integer.

    Args:
        int_string (str): The string representation of the integer.

    Returns:
        int: The integer value of the string.

    Raises:
        ValueError: If the string is not a valid integer.
    """
    try:
        return int(int_string)
    except ValueError:
        raise ValueError("Invalid integer string")


def str_to_float(float_string: str) -> float:
    """
    Converts a string to a float.

    Args:
        float_string (str): The string representation of the float.

    Returns:
        float: The float value of the string.

    Raises:
        ValueError: If the string is not a valid float.
    """
    try:
        return float(float_string)
    except ValueError:
        raise ValueError("Invalid float string")
