# utils/logger.py

import logging
import sys
from logging import Logger


def get_logger(name: str) -> Logger:
    """
    Initializes and returns a configured logger.

    This function creates a logger with a specified name, sets its level to INFO,
    and attaches a handler that prints log messages to standard output. If a
    logger with the same name has already been configured, it returns the
    existing logger instance.

    Args:
        name: The name of the logger, typically __name__.

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
