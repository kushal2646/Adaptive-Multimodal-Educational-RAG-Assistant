"""
utils/logger.py
===============
Structured logging — Windows cp1252 safe (no emoji in log strings).
"""

import logging
import sys

try:
    from rich.logging import RichHandler
    from rich.console import Console
    # Test if the console can handle unicode
    _console = Console(force_terminal=True, highlight=False)
    _USE_RICH = True
except Exception:
    _USE_RICH = False


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger, falling back to standard logging on Windows."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        try:
            from config.settings import LOG_LEVEL
        except Exception:
            LOG_LEVEL = "INFO"

        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

        if _USE_RICH:
            try:
                handler = RichHandler(
                    console=_console,
                    rich_tracebacks=False,
                    markup=False,
                    show_path=False,
                )
                handler.setLevel(level)
                formatter = logging.Formatter("%(message)s", datefmt="[%X]")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                return logger
            except Exception:
                pass

        # Fallback: plain StreamHandler (safe on Windows cp1252)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
