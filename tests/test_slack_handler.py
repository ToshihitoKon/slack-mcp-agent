"""ThreadLockManager の直列化・即時解放挙動、および空応答ガードを検証する。"""

import asyncio

import pytest

from slack_agent.slack_handler import (
    _EMPTY_RESPONSE_MESSAGE,
    _REFUSAL_MESSAGE,
    _ensure_non_empty_answer,
    ThreadLockManager,
)


@pytest.mark.asyncio
async def test_same_key_serializes_critical_sections():
    """同一 key の critical section は直列化される。"""
    manager = ThreadLockManager()
    order: list[str] = []

    async def worker(tag: str, hold: float):
        lock = await manager.acquire("thread-1")
        try:
            async with lock:
                order.append(f"{tag}-enter")
                await asyncio.sleep(hold)
                order.append(f"{tag}-exit")
        finally:
            await manager.release("thread-1")

    # A が先に走り出し、B は A の exit 後にしか enter できない
    await asyncio.gather(worker("A", 0.05), worker("B", 0.0))

    assert order == ["A-enter", "A-exit", "B-enter", "B-exit"]


@pytest.mark.asyncio
async def test_distinct_keys_run_concurrently():
    """異なる key の処理は並行に進む。"""
    manager = ThreadLockManager()
    in_section = 0
    max_concurrent = 0

    async def worker(key: str):
        nonlocal in_section, max_concurrent
        lock = await manager.acquire(key)
        try:
            async with lock:
                in_section += 1
                max_concurrent = max(max_concurrent, in_section)
                await asyncio.sleep(0.02)
                in_section -= 1
        finally:
            await manager.release(key)

    await asyncio.gather(worker("A"), worker("B"))

    # 別 thread なので同時に critical section に入れる
    assert max_concurrent == 2


@pytest.mark.asyncio
async def test_lock_is_released_after_use():
    """acquire / release 後に lock が破棄される (即時解放)。"""
    manager = ThreadLockManager()

    lock = await manager.acquire("thread-1")
    assert manager.active_count() == 1
    async with lock:
        pass
    await manager.release("thread-1")

    assert manager.active_count() == 0


@pytest.mark.asyncio
async def test_lock_survives_while_waiters_exist():
    """待機者がいる間は lock が破棄されない。"""
    manager = ThreadLockManager()

    # 1 人目が保持中
    lock1 = await manager.acquire("thread-1")
    await lock1.acquire()

    # 2 人目が acquire (lock 取得待ち)。参照カウントは 2 になる
    lock2 = await manager.acquire("thread-1")
    assert manager.active_count() == 1
    # 同じ lock オブジェクトが共有される
    assert lock1 is lock2

    # 1 人目が release しても 2 人目が残っているので破棄されない
    lock1.release()
    await manager.release("thread-1")
    assert manager.active_count() == 1

    # 2 人目が抜けて初めて破棄される
    await manager.release("thread-1")
    assert manager.active_count() == 0


@pytest.mark.asyncio
async def test_lock_released_even_on_exception():
    """critical section で例外が出ても lock は解放される。"""
    manager = ThreadLockManager()

    async def failing():
        lock = await manager.acquire("thread-1")
        try:
            async with lock:
                raise RuntimeError("boom")
        finally:
            await manager.release("thread-1")

    with pytest.raises(RuntimeError, match="boom"):
        await failing()

    assert manager.active_count() == 0
    # 解放後に再取得しても問題ない
    lock = await manager.acquire("thread-1")
    async with lock:
        pass
    await manager.release("thread-1")
    assert manager.active_count() == 0


@pytest.mark.asyncio
async def test_release_unknown_key_is_safe():
    """未知 key の release を呼んでもエラーにならず負にもならない。"""
    manager = ThreadLockManager()
    await manager.release("never-acquired")
    assert manager.active_count() == 0


def test_ensure_non_empty_answer_passes_through_non_empty_text():
    """通常の応答はそのまま返す。"""
    assert _ensure_non_empty_answer("こんにちは", thread_ts="100") == "こんにちは"


@pytest.mark.parametrize("answer", ["", "   ", "\n"])
def test_ensure_non_empty_answer_falls_back_on_empty_text(answer):
    """空文字列や空白のみの応答はフォールバック文言に置き換える (no_text エラー防止)。"""
    assert _ensure_non_empty_answer(answer, thread_ts="100") == _EMPTY_RESPONSE_MESSAGE


def test_ensure_non_empty_answer_uses_refusal_message_on_refusal_stop_reason():
    """stop_reason=refusal は Claude の安全性分類器による拒否なので、
    再試行を促す通常の空応答メッセージとは別の専用文言を返す。"""
    answer = _ensure_non_empty_answer("", thread_ts="100", stop_reason="refusal")
    assert answer == _REFUSAL_MESSAGE
    assert answer != _EMPTY_RESPONSE_MESSAGE
