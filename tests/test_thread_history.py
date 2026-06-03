"""fetch_new_replies の差分取得・フィルタリング挙動を検証する。"""

import pytest
from langchain_core.messages import HumanMessage

from slack_agent.thread_history import fetch_new_replies

BOT_ID = "UBOT"


class _FakeClient:
    """conversations_replies の応答を差し替えるモック client。"""

    def __init__(self, messages, raise_exc: Exception | None = None):
        self._messages = messages
        self._raise = raise_exc
        self.last_kwargs = None

    async def conversations_replies(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raise is not None:
            raise self._raise
        return {"messages": self._messages}


@pytest.mark.asyncio
async def test_returns_empty_on_api_error():
    client = _FakeClient([], raise_exc=RuntimeError("api down"))
    result = await fetch_new_replies(
        client, channel="C1", thread_root_ts="100", since_ts="50",
        current_ts="200", bot_user_id=BOT_ID,
    )
    assert result == []


@pytest.mark.asyncio
async def test_diff_mode_sets_oldest_and_inclusive():
    client = _FakeClient([])
    await fetch_new_replies(
        client, channel="C1", thread_root_ts="100", since_ts="50",
        current_ts="200", bot_user_id=BOT_ID,
    )
    assert client.last_kwargs["oldest"] == "50"
    assert client.last_kwargs["inclusive"] is False


@pytest.mark.asyncio
async def test_fallback_mode_omits_oldest():
    client = _FakeClient([])
    await fetch_new_replies(
        client, channel="C1", thread_root_ts="100", since_ts=None,
        current_ts="200", bot_user_id=BOT_ID,
    )
    assert "oldest" not in client.last_kwargs
    assert "inclusive" not in client.last_kwargs


@pytest.mark.asyncio
async def test_skips_subtype_and_current_ts_and_empty():
    msgs = [
        {"ts": "110", "user": "U1", "subtype": "channel_join", "text": "joined"},
        {"ts": "200", "user": "U1", "text": "current message"},  # current_ts
        {"ts": "120", "user": "U1", "text": "   "},  # 空白のみ
        {"ts": "130", "user": "U1", "text": "valid"},
    ]
    client = _FakeClient(msgs)
    result = await fetch_new_replies(
        client, channel="C1", thread_root_ts="100", since_ts="50",
        current_ts="200", bot_user_id=BOT_ID,
    )
    assert [str(m.content) for m in result] == ["valid"]


@pytest.mark.asyncio
async def test_diff_mode_excludes_thread_root():
    msgs = [
        {"ts": "100", "user": "U1", "text": "root message"},  # diff では除外
        {"ts": "130", "user": "U1", "text": "reply"},
    ]
    client = _FakeClient(msgs)
    result = await fetch_new_replies(
        client, channel="C1", thread_root_ts="100", since_ts="50",
        current_ts="999", bot_user_id=BOT_ID,
    )
    assert [str(m.content) for m in result] == ["reply"]


@pytest.mark.asyncio
async def test_fallback_mode_includes_thread_root():
    msgs = [
        {"ts": "100", "user": "U1", "text": "root message"},  # fallback では取込
        {"ts": "130", "user": "U1", "text": "reply"},
    ]
    client = _FakeClient(msgs)
    result = await fetch_new_replies(
        client, channel="C1", thread_root_ts="100", since_ts=None,
        current_ts="999", bot_user_id=BOT_ID,
    )
    assert [str(m.content) for m in result] == ["root message", "reply"]


@pytest.mark.asyncio
async def test_strips_mentions():
    msgs = [{"ts": "130", "user": "U1", "text": "<@UBOT> hello there"}]
    client = _FakeClient(msgs)
    result = await fetch_new_replies(
        client, channel="C1", thread_root_ts="100", since_ts="50",
        current_ts="999", bot_user_id=BOT_ID,
    )
    assert [str(m.content) for m in result] == ["hello there"]


@pytest.mark.asyncio
async def test_assistant_messages_get_prefix():
    msgs = [
        {"ts": "130", "user": BOT_ID, "text": "I am the bot"},
        {"ts": "140", "bot_id": "B123", "text": "from a bot integration"},
        {"ts": "150", "user": "U1", "text": "human reply"},
    ]
    client = _FakeClient(msgs)
    result = await fetch_new_replies(
        client, channel="C1", thread_root_ts="100", since_ts="50",
        current_ts="999", bot_user_id=BOT_ID,
    )
    contents = [str(m.content) for m in result]
    assert contents == [
        "[assistant said]: I am the bot",
        "[assistant said]: from a bot integration",
        "human reply",
    ]
    assert all(isinstance(m, HumanMessage) for m in result)
