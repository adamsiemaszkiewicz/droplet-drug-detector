# -*- coding: utf-8 -*-

import ast
from pathlib import Path
from typing import Dict, List


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
    if isinstance(dict_string, Dict):
        return dict_string
    try:
        return ast.literal_eval(dict_string)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid dictionary string")


def str_to_list(list_string: str) -> List:
    """
    Converts a string representation of a list into a list object.

    The string should be in a format interpretable as a Python list literal,
    e.g., "[1, 2, 3]" or "['a', 'b', 'c']".

    Args:
        list_string (str): The string representation of the list.

    Returns:
        List: A list object parsed from the string.

    Raises:
        ValueError: If the string is not a valid list representation.
    """
    if isinstance(list_string, List):
        return list_string
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
    if isinstance(bool_string, bool):
        return bool_string

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
    if isinstance(int_string, int):
        return int_string
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
    if isinstance(float_string, float):
        return float_string
    try:
        return float(float_string)
    except ValueError:
        raise ValueError("Invalid float string")


def path_to_str(path: Path) -> str:
    """
    Converts a Path object to its string representation.

    Args:
        path (Path): The Path object to convert.

    Returns:
        str: The string representation of the Path object.
    """
    return path.as_posix() if path else ""
