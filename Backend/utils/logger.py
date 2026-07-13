"""
Mosaic Logging System

Centralized logging configuration with:
- File logging (mosaic.log) — all levels, rotates at 10MB
- Console logging — INFO+ with colored output
- Structured format with timestamps, module, and level
- Separate error log (mosaic_errors.log) for quick diagnosis
"""

import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file paths
MAIN_LOG = os.path.join(LOG_DIR, "mosaic.log")
ERROR_LOG = os.path.join(LOG_DIR, "mosaic_errors.log")
REQUEST_LOG = os.path.join(LOG_DIR, "requests.log")


class ColorFormatter(logging.Formatter):
    """Colored console output for different log levels."""
    
    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure the root Mosaic logger.
    
    Returns the 'mosaic' logger that all modules should use.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root mosaic logger
    logger = logging.getLogger("mosaic")
    logger.setLevel(logging.DEBUG)  # Capture everything, handlers filter
    logger.propagate = False

    # Clear existing handlers (for reload safety)
    logger.handlers.clear()

    # --- File handler: all logs, rotating ---
    file_handler = RotatingFileHandler(
        MAIN_LOG, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    # --- Error file handler: errors only ---
    error_handler = RotatingFileHandler(
        ERROR_LOG, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s\n"
        "  → %(pathname)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(error_handler)

    # --- Console handler: colored, INFO+ ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColorFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the 'mosaic' namespace.
    
    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    return logging.getLogger(f"mosaic.{name}")


def get_request_logger() -> logging.Logger:
    """
    Get a dedicated logger for HTTP request/response tracking.
    Writes to a separate requests.log file.
    """
    logger = logging.getLogger("mosaic.requests")
    
    if not logger.handlers:
        handler = RotatingFileHandler(
            REQUEST_LOG, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger
