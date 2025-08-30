"""Log level enumeration for the inference framework."""

import logging
from enum import Enum


class LogLevel(Enum):
    """Enumeration of logging levels for the inference framework.

    Attributes:
        CRITICAL (str): Critical errors that cause the program to stop.
        ERROR (str): Errors that do not require immediate program termination.
        WARNING (str): Warnings about potential problems.
        INFO (str): General informational messages.
        DEBUG (str): Detailed debugging information.
        NOTSET (str): No specific logging level set.
    """

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"

    def to_logging_level(self):
        """Convert the enum value to the corresponding logging module level.

        Returns:
            int: The logging module level (e.g., logging.INFO).
        """
        return getattr(logging, self.value, logging.INFO)
