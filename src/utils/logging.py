"""Logger setup utility for the inference framework."""

import logging
import sys
import os
from datetime import datetime
from utils.log_levels import LogLevel

def setup_logger(name="inference_framework", log_file=None, level=None):
    """Create and configure a standard logger for the inference framework.

    The logging level is taken from the 'LOG_LEVEL' environment variable if not explicitly provided.
    The log file name includes the current date by default.

    Args:
        name (str, optional): Name of the logger instance. Defaults to "inference_framework".
        log_file (str, optional): Path to the log file. If None, a default file with the 
            current date is used.
        level (int or LogLevel, optional): Logging level (e.g., logging.INFO, logging.DEBUG, 
            LogLevel.INFO).
            If None, uses LOG_LEVEL environment variable or INFO.
            
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Determine log file name
    if log_file is None:
        today = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
        log_file = f"logs/experiments-{today}.log"

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Determine log level
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
    elif isinstance(level, LogLevel):
        level = level.to_logging_level()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if logger is recreated in the same process
    if not logger.handlers:
        # File log handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        file_handler.setFormatter(file_format)

        # Console log handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_format)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


    return logger

# Global logger instance: respects LOG_LEVEL env or default INFO, log file named by date
logger = setup_logger()
