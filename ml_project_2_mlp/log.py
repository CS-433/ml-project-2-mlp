"""
Simple module that enables easy global logging.

Includes:
    - setup_logging: Basic logging configuration.
    - get_logger: Returns a logger for the given name.
"""

import logging


def setup_logging(level=logging.INFO):
    """
    Sets the global logging level (only display messages with level >= level).

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        None
    """
    # Example: [INFO] (log.setup_logging) - Setting up logging
    format = "[%(levelname)s] (%(name)s.%(funcName)s) - %(message)s"
    logging.basicConfig(level=level, format=format)


def get_logger(name: str) -> logging.Logger:
    """
    Gets a logger for the given name.
    """
    return logging.getLogger(name)
