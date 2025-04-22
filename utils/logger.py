import logging
import os
import sys
from CONSTANTS import LOG_ROOT, SESSION

def setup_logger():
    """Set up and configure logger"""
    # Create logger
    logger = logging.getLogger("MTALog")
    logger.setLevel(logging.INFO)
    
    # Remove all existing handlers to start fresh
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create file handler which logs even debug messages
    if not os.path.exists(LOG_ROOT):
        os.makedirs(LOG_ROOT)
    
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "mtalog.log"))
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Custom filter to only show evaluation metrics
    class EvaluationFilter(logging.Filter):
        def filter(self, record):
            return "Evaluation metrics" in record.getMessage() or "F1=" in record.getMessage()
    
    # Apply filter to console handler only
    eval_filter = EvaluationFilter()
    console_handler.addFilter(eval_filter)
    
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger