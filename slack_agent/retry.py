import asyncio
import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that should not be retried
_NO_RETRY_STATUS_CODES = frozenset(range(400, 500))


def _should_retry_exception(exc: Exception) -> bool:
    """Return True if the exception is retryable."""
    # Check for HTTP status in exception attributes (httpx, aiohttp, requests style)
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status_code is not None and status_code in _NO_RETRY_STATUS_CODES:
        return False
    return True


async def retry_async(
    func: Callable,
    *args,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    **kwargs,
):
    """Run an async callable with exponential backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _should_retry_exception(exc):
                logger.warning("Non-retryable error (attempt %d/%d): %s", attempt, max_attempts, exc)
                raise
            if attempt == max_attempts:
                break
            wait = backoff_base * (2 ** (attempt - 1))
            logger.warning(
                "Retryable error (attempt %d/%d), retrying in %.1fs: %s",
                attempt, max_attempts, wait, exc,
            )
            await asyncio.sleep(wait)
    raise last_exc
