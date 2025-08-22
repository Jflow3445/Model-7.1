# utils/logging_utils.py
from __future__ import annotations
import datetime
import logging
import os
import sys
import json
from logging.handlers import RotatingFileHandler
from typing import Optional
from config.settings import BASE_DIR, LOG_LEVEL, TIMEZONE
import pytz

# ======== CONFIGURABLE PARAMETERS ========
LOG_FILE = os.getenv(
    "FOREX_LOG_FILE",
    os.path.join(BASE_DIR, "forex_training.log")
)
MAX_LOG_FILE_SIZE = int(os.getenv("FOREX_LOG_MAX_SIZE", 10 * 1024 * 1024))
BACKUP_COUNT = int(os.getenv("FOREX_LOG_BACKUPS", 5))
LOG_FORMAT = os.getenv(
    "FOREX_LOG_FORMAT",
    "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

# ======== FORMATTERS ========
class TZFormatter(logging.Formatter):
    """Time-zone aware formatter using TIMEZONE from config."""
    def formatTime(self, record, datefmt=None):
        tz = pytz.timezone(TIMEZONE)
        ct = datetime.datetime.fromtimestamp(record.created, tz)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime("%Y-%m-%dT%H:%M:%S%z")

class JSONFormatter(TZFormatter):
    """Outputs logs in JSON format."""
    def format(self, record):
        obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "file": record.pathname,
            "line": record.lineno,
        }
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj)

# ======== LOGGER SETUP ========
logger = logging.getLogger("forex")
logger.setLevel(LOG_LEVEL)
# clear old handlers
for handler in list(logger.handlers):
    logger.removeHandler(handler)

# console handler
console = logging.StreamHandler(sys.stdout)
console.setLevel(LOG_LEVEL)
console.setFormatter(TZFormatter(LOG_FORMAT))
logger.addHandler(console)

# rotating file handler
fileh = RotatingFileHandler(
    LOG_FILE, maxBytes=MAX_LOG_FILE_SIZE, backupCount=BACKUP_COUNT
)
fileh.setLevel(LOG_LEVEL)
fileh.setFormatter(JSONFormatter())
logger.addHandler(fileh)

# ======== API ========

def log_event(message: str, level: str = "info") -> None:
    """
    Log a message at the specified level. Supported: debug, info, warning, error, critical.
    """
    lvl = level.lower()
    if lvl == "debug":
        logger.debug(message)
    elif lvl == "info":
        logger.info(message)
    elif lvl == "warning":
        logger.warning(message)
    elif lvl == "error":
        logger.error(message)
    elif lvl == "critical":
        logger.critical(message)
    else:
        logger.info(message)


def log_exception(message: str, exc_info: bool = True) -> None:
    """
    Log an exception with optional traceback.
    """
    logger.error(message, exc_info=exc_info)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a named logger with the same handlers as the root 'forex' logger.
    """
    return logging.getLogger(name or "forex")
