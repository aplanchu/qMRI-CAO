"""
Logging levels:
CRITICAL    50
ERROR       40
WARNING     30
INFO        20
DEBUG       10
NOTSET      0
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOGGER_NAME = "MUDI"
FORMATTER = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
)


class BraceString(str):
    def __mod__(self, other):
        return self.format(*other)

    def __str__(self):
        return self


class StyleAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super(StyleAdapter, self).__init__(logger, extra)

    def process(self, msg, kwargs):
        if kwargs.pop("style", "%") == "{":  # optional
            msg = BraceString(msg)
        return msg, kwargs


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + FORMATTER + reset,
        logging.INFO: grey + FORMATTER + reset,
        logging.WARNING: yellow + FORMATTER + reset,
        logging.ERROR: red + FORMATTER + reset,
        logging.CRITICAL: bold_red + FORMATTER + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger() -> None:
    """Create a logger with a stream handler and a file handler"""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.hasHandlers():  # only do this once.
        log_level = logger.getEffectiveLevel()

        path = Path(Path(__file__).cwd(), "logs", f"{LOGGER_NAME}.log")
        path.parent.mkdir(parents=True, exist_ok=True)

        streamHandler = logging.StreamHandler(stream=sys.stdout)
        streamHandler.setLevel(log_level)
        streamHandler.setFormatter(ColorFormatter())

        fileHandler = TimedRotatingFileHandler(path, when="h", interval=1)
        fileHandler.setLevel(log_level)
        fileHandler.setFormatter(logging.Formatter(FORMATTER))

        logger.addHandler(streamHandler)
        logger.addHandler(fileHandler)

        logger = StyleAdapter(logger)

    return logger


def set_log_level(log_level: int) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)
    logger.handlers = []
    get_logger()


logger = get_logger()
