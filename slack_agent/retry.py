import asyncio
import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that should not be retried
_NO_RETRY_STATUS_CODES = frozenset(range(400, 500))

# リトライしても無駄なタイムアウト系の例外クラス名。
# MCP ツールの応答待ちタイムアウト (McpError "Timed out ... Waited N seconds") は
# 同じ呼び出しを再試行しても同じ時間待たされるだけで、サーバ側が無応答な限り
# 成功しない。リトライ対象外にしてロック占有と待ち時間を減らす。
_NO_RETRY_EXC_NAMES = frozenset({"McpError", "TimeoutError"})


def _should_retry_exception(exc: Exception) -> bool:
    """Return True if the exception is retryable."""
    # Check for HTTP status in exception attributes (httpx, aiohttp, requests style)
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status_code is not None and status_code in _NO_RETRY_STATUS_CODES:
        return False
    # タイムアウト系は再試行しても無駄なので即座に失敗させる。
    if type(exc).__name__ in _NO_RETRY_EXC_NAMES:
        return False
    return True


def _describe_exception(exc: BaseException) -> str:
    """例外を説明する文字列。ExceptionGroup (TaskGroup 例外) は
    sub-exception を再帰的に展開して中身を見えるようにする。
    """
    group = getattr(exc, "exceptions", None)
    if group is not None:
        inner = "; ".join(_describe_exception(e) for e in group)
        return f"{type(exc).__name__}[{inner}]"
    return f"{type(exc).__name__}: {exc}"


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
            desc = _describe_exception(exc)
            if not _should_retry_exception(exc):
                logger.warning("Non-retryable error (attempt %d/%d): %s", attempt, max_attempts, desc)
                raise
            if attempt == max_attempts:
                break
            wait = backoff_base * (2 ** (attempt - 1))
            logger.warning(
                "Retryable error (attempt %d/%d), retrying in %.1fs: %s",
                attempt, max_attempts, wait, desc,
            )
            await asyncio.sleep(wait)
    logger.error(
        "All %d attempts failed: %s", max_attempts, _describe_exception(last_exc)
    )
    raise last_exc
