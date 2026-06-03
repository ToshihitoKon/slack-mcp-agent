"""MCP ツールラッパーの挙動を検証する。

主目的:
- call_tool への read_timeout_seconds 伝播 (応答が返らない MCP サーバーで
  無限ハングしないため)。
- 保持したセッションを使い回し、ツール呼び出しごとにサーバーを起動し直さない
  こと (Issue #6 の再起動レース対策)。
"""

import asyncio
from datetime import timedelta

import pytest

from slack_agent.tools import _make_langchain_tool


class _FakeMcpTool:
    def __init__(self, name="search"):
        self.name = name
        self.description = "desc"
        self.inputSchema = {"type": "object", "properties": {}}


class _FakeResult:
    content = "tool output"


class _FakeSession:
    """ClientSession のモック。call_tool の呼び出しを記録する。"""

    def __init__(self):
        self.calls: list[dict] = []

    async def call_tool(self, name, arguments=None, read_timeout_seconds=None, **kw):
        self.calls.append(
            {
                "name": name,
                "arguments": arguments,
                "read_timeout_seconds": read_timeout_seconds,
            }
        )
        return _FakeResult()


def _make_tool(mcp_tool=None, session=None, timeout=None):
    session = session or _FakeSession()
    tool = _make_langchain_tool(
        "srv", mcp_tool or _FakeMcpTool(), session, asyncio.Lock(), timeout
    )
    return tool, session


@pytest.mark.asyncio
async def test_tool_passes_read_timeout_as_timedelta():
    tool, session = _make_tool(timeout=30.0)
    result = await tool.ainvoke({"q": "hello"})

    assert result == "tool output"
    call = session.calls[0]
    assert call["name"] == "search"
    assert call["arguments"] == {"q": "hello"}
    assert call["read_timeout_seconds"] == timedelta(seconds=30.0)


@pytest.mark.asyncio
async def test_tool_no_timeout_passes_none():
    tool, session = _make_tool(timeout=None)
    await tool.ainvoke({"q": "hello"})
    assert session.calls[0]["read_timeout_seconds"] is None


@pytest.mark.asyncio
async def test_tool_name_is_prefixed_with_server():
    tool, _ = _make_tool(mcp_tool=_FakeMcpTool(name="fetch"))
    assert tool.name == "srv__fetch"


@pytest.mark.asyncio
async def test_tool_reuses_same_session_across_calls():
    """複数回呼んでも同じ保持セッションを使い、起動し直さない。"""
    tool, session = _make_tool()
    await tool.ainvoke({"q": "a"})
    await tool.ainvoke({"q": "b"})

    # 同一セッションに 2 回 call_tool されている (再接続していない)
    assert len(session.calls) == 2
    assert [c["arguments"] for c in session.calls] == [{"q": "a"}, {"q": "b"}]
