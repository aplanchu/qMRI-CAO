from pathlib import Path
from argparse import ArgumentTypeError


def dir_path(path: str) -> Path:
    """Check if path is a directory. Used as argparse type.

    Args:
        path (str): path to check

    Raises:
        ArgumentTypeError: raised if path is not a directory

    Returns:
        Path: constructed Path
    """
    path = Path(path)
    if path.is_dir():
        return path
    else:
        raise ArgumentTypeError(f"argument value {path} is not a directory")


def file_path(path: str) -> Path:
    """Check if path is a file. Used as argparse type.

    Args:
        path (str): path to check

    Raises:
        ArgumentTypeError: raised if path is not a file.

    Returns:
        Path: constructed Path
    """
    path = Path(path)
    if path.is_file():
        return path
    else:
        raise ArgumentTypeError(f"argument value {path} is not a file")
