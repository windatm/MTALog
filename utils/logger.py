
import logging
import sys
import os
from CONSTANTS import LOG_ROOT, SESSION

def setup_logger(name="MTALog", log_file="MTALog.log", level=logging.DEBUG):
    """
    Set up a logger with console and file handlers.

    Args:
        name (str): Name of the logger.
        log_file (str): Log file name (inside LOG_ROOT).
        level (int): Logging level (default: DEBUG).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        f"%(asctime)s - %(name)s - {SESSION} - %(levelname)s: %(message)s"
    )

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, log_file))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logger for {name} constructed successfully. Current working directory: {os.getcwd()}. Logs will be written in {LOG_ROOT}.")
    return logger