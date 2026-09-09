from __future__ import annotations

import logging


LOGGER_NAME = "contextual_crc"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure simple console logging and return the project logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(numeric_level)
    return logger
