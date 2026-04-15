"""
decorators.py — Reusable function decorators for the trading bot.
"""

from __future__ import annotations

import functools
import time
from typing import Callable, Type

from ..logger import get_logger

log = get_logger(__name__)


def retry(
    exceptions: tuple = (Exception,),
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    logger_name: str = __name__,
):
    """Retry a function on specified exceptions with exponential back-off.

    Args:
        exceptions: Tuple of exception types to catch and retry on.
        max_attempts: Maximum number of attempts (including the first).
        base_delay: Seconds to wait before the first retry.
        backoff: Multiplier applied to the delay after each failure.
        logger_name: Name of the logger to use for retry warnings.

    Returns:
        Decorated function.
    """
    _log = get_logger(logger_name)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        _log.error(
                            "%s failed after %d attempts: %s",
                            func.__qualname__,
                            max_attempts,
                            exc,
                        )
                        raise
                    _log.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.1fs.",
                        func.__qualname__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


def timed(func: Callable) -> Callable:
    """Log the wall-clock execution time of a function.

    Args:
        func: The function to wrap.

    Returns:
        Decorated function that logs its runtime at DEBUG level.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log.debug("%s completed in %.3fs.", func.__qualname__, elapsed)
        return result
    return wrapper


def singleton(cls: Type) -> Type:
    """Class decorator that restricts instantiation to a single instance.

    Args:
        cls: The class to make into a singleton.

    Returns:
        The class, wrapped so subsequent instantiations return the same object.
    """
    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
