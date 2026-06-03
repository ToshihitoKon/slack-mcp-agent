"""retry_async / _should_retry_exception の挙動を検証する。"""

import pytest

from slack_agent.retry import _should_retry_exception, retry_async


class _HTTPError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _StatusError(Exception):
    """status 属性 (aiohttp 風) を持つ例外。"""

    def __init__(self, status: int):
        super().__init__(f"status {status}")
        self.status = status


def test_should_retry_4xx_is_not_retryable():
    assert _should_retry_exception(_HTTPError(404)) is False
    assert _should_retry_exception(_HTTPError(400)) is False
    assert _should_retry_exception(_HTTPError(499)) is False


def test_should_retry_5xx_is_retryable():
    assert _should_retry_exception(_HTTPError(500)) is True
    assert _should_retry_exception(_HTTPError(503)) is True


def test_should_retry_status_attr_is_respected():
    assert _should_retry_exception(_StatusError(404)) is False
    assert _should_retry_exception(_StatusError(500)) is True


def test_should_retry_plain_exception_is_retryable():
    assert _should_retry_exception(ValueError("boom")) is True


@pytest.mark.asyncio
async def test_retry_returns_on_success():
    calls = []

    async def func():
        calls.append(1)
        return "ok"

    result = await retry_async(func, max_attempts=3, backoff_base=0)
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retry_recovers_after_transient_failures():
    calls = []

    async def func():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return "recovered"

    result = await retry_async(func, max_attempts=3, backoff_base=0)
    assert result == "recovered"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_retry_raises_last_exception_after_max_attempts():
    calls = []

    async def func():
        calls.append(1)
        raise RuntimeError(f"fail {len(calls)}")

    with pytest.raises(RuntimeError, match="fail 2"):
        await retry_async(func, max_attempts=2, backoff_base=0)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_retry_does_not_retry_non_retryable():
    calls = []

    async def func():
        calls.append(1)
        raise _HTTPError(400)

    with pytest.raises(_HTTPError):
        await retry_async(func, max_attempts=5, backoff_base=0)
    # 4xx は即座に raise され、リトライされない
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retry_passes_through_args_and_kwargs():
    async def func(a, b, *, c):
        return a + b + c

    result = await retry_async(func, 1, 2, max_attempts=1, backoff_base=0, c=3)
    assert result == 6
