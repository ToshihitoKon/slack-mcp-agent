"""MCP Result cache のデータフローを検証するテスト。

tool_executor_node が決定的に cache_key を付与し、compressor_node が
それを使って cache_store に保存・CacheReference に登録する一連の流れを確認する。
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from slack_agent.cache import CacheStore, InMemoryCacheStore
from slack_agent.nodes import compressor_node, tool_executor_node


class _RetryConfig:
    max_attempts = 1
    backoff_base_seconds = 0


class _DummyTool:
    """常に固定サイズの結果を返すツール。"""

    def __init__(self, size: int):
        self._size = size

    async def ainvoke(self, args):
        return "X" * self._size


class _FakeLightLLM:
    """compressor が呼ぶ light_llm のモック。cache_key は返さない。"""

    async def ainvoke(self, messages):
        return AIMessage(content='{"focused_summary": "sum", "content_index": "idx"}')


class _BrokenLightLLM:
    """不正な JSON を返す light_llm のモック。"""

    async def ainvoke(self, messages):
        return AIMessage(content="not a json")


def _ai_with_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.mark.asyncio
async def test_tool_executor_attaches_deterministic_cache_key():
    """tool 実行時に make_key と一致する cache_key が ToolMessage に付与される。"""
    ai = _ai_with_tool_call("srv__search", {"q": "hello"}, "tc1")
    state = {"messages": [ai]}

    out = await tool_executor_node(
        state, {"srv__search": _DummyTool(100)}, None, _RetryConfig()
    )

    tm = out["messages"][0]
    expected_key = CacheStore.make_key("srv__search", {"q": "hello"})
    assert tm.additional_kwargs["cache_key"] == expected_key
    assert tm.additional_kwargs["cache_tool_name"] == "srv__search"
    assert tm.additional_kwargs["cache_tool_args"] == {"q": "hello"}


@pytest.mark.asyncio
async def test_cache_fetcher_result_is_not_cached():
    """cache_fetcher 自身の結果はキャッシュ対象外 (cache_key を付けない)。"""
    ai = _ai_with_tool_call("cache_fetcher", {"cache_key": "k"}, "tc2")
    state = {"messages": [ai]}

    out = await tool_executor_node(
        state, {"cache_fetcher": _DummyTool(100)}, None, _RetryConfig()
    )

    assert out["messages"][0].additional_kwargs == {}


@pytest.mark.asyncio
async def test_compressor_saves_to_cache_using_tool_metadata():
    """compressor が ToolMessage のメタから cache_key を取り保存・参照登録する。"""
    store = InMemoryCacheStore()
    expected_key = CacheStore.make_key("srv__search", {"q": "hello"})

    # 閾値 (10000) を超える大きい結果を持つ ToolMessage
    big_result = "X" * 20000
    tm = ToolMessage(
        content=big_result,
        tool_call_id="tc1",
        additional_kwargs={
            "cache_key": expected_key,
            "cache_tool_name": "srv__search",
            "cache_tool_args": {"q": "hello"},
        },
    )
    state = {
        "messages": [tm],
        "compression_threshold": 10000,
        "cache_references": [],
    }

    out = await compressor_node(state, _FakeLightLLM(), store, 24, _RetryConfig())

    # CacheReference が正しく登録される
    refs = out["cache_references"]
    assert len(refs) == 1
    assert refs[0]["cache_key"] == expected_key
    assert refs[0]["tool_name"] == "srv__search"
    assert refs[0]["tool_args"] == {"q": "hello"}
    assert refs[0]["content_index"] == "idx"

    # cache_store から raw_result が取得できる
    entry = store.get(expected_key)
    assert entry is not None
    assert entry.raw_result == big_result

    # メッセージは要約に置換されている (id を引き継ぐ)
    compressed = out["messages"][0]
    assert compressed.content.startswith("[Compressed]")


@pytest.mark.asyncio
async def test_small_result_is_not_compressed_or_cached():
    """閾値以下の結果は圧縮もキャッシュもされない。"""
    store = InMemoryCacheStore()
    key = CacheStore.make_key("srv__search", {"q": "hi"})
    tm = ToolMessage(
        content="small",
        tool_call_id="tc1",
        additional_kwargs={
            "cache_key": key,
            "cache_tool_name": "srv__search",
            "cache_tool_args": {"q": "hi"},
        },
    )
    state = {
        "messages": [tm],
        "compression_threshold": 10000,
        "cache_references": [],
    }

    out = await compressor_node(state, _FakeLightLLM(), store, 24, _RetryConfig())

    assert out["cache_references"] == []
    assert store.get(key) is None
    assert out["messages"][0].content == "small"


@pytest.mark.asyncio
async def test_compressor_keeps_original_message_on_invalid_json():
    """compressor LLM が不正な JSON を返したら元の ToolMessage を保持しキャッシュしない。"""
    store = InMemoryCacheStore()
    key = CacheStore.make_key("srv__search", {"q": "hello"})
    big_result = "X" * 20000
    tm = ToolMessage(
        content=big_result,
        tool_call_id="tc1",
        additional_kwargs={
            "cache_key": key,
            "cache_tool_name": "srv__search",
            "cache_tool_args": {"q": "hello"},
        },
    )
    state = {
        "messages": [tm],
        "compression_threshold": 10000,
        "cache_references": [],
    }

    out = await compressor_node(state, _BrokenLightLLM(), store, 24, _RetryConfig())

    # 圧縮失敗時は元メッセージをそのまま残し、キャッシュもしない
    assert out["messages"][0].content == big_result
    assert out["cache_references"] == []
    assert store.get(key) is None


@pytest.mark.asyncio
async def test_non_tool_messages_pass_through_compressor():
    """ToolMessage 以外のメッセージは compressor を素通りする。"""
    store = InMemoryCacheStore()
    ai = AIMessage(content="some answer")
    state = {
        "messages": [ai],
        "compression_threshold": 10000,
        "cache_references": [],
    }

    out = await compressor_node(state, _FakeLightLLM(), store, 24, _RetryConfig())

    assert out["messages"][0] is ai
    assert out["cache_references"] == []
