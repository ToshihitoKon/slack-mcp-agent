"""MCP ツールラッパーの挙動を検証する。

主目的は call_tool への read_timeout_seconds 伝播 (応答が返らない MCP
サーバーで無限ハングしないため)。stdio_client / ClientSession をモックして
_arun が call_tool に渡す引数を捕捉する。
"""

import sys
import types
from datetime import timedelta

import pytest

from slack_agent import tools as tools_mod
from slack_agent.tools import _make_langchain_tool


class _FakeMcpTool:
    def __init__(self, name="search"):
        self.name = name
        self.description = "desc"
        self.inputSchema = {"type": "object", "properties": {}}


class _FakeResult:
    content = "tool output"


class _FakeSession:
    """ClientSession のモック。call_tool の引数を記録する。"""

    last_call_kwargs: dict = {}

    def __init__(self, read, write):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return None

    async def call_tool(self, name, arguments=None, read_timeout_seconds=None, **kw):
        _FakeSession.last_call_kwargs = {
            "name": name,
            "arguments": arguments,
            "read_timeout_seconds": read_timeout_seconds,
        }
        return _FakeResult()


class _FakeStdioClient:
    """stdio_client(params) のモック (async context manager)。"""

    def __init__(self, params):
        pass

    async def __aenter__(self):
        return ("read", "write")

    async def __aexit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _patch_mcp(monkeypatch):
    monkeypatch.setattr(tools_mod, "stdio_client", lambda params: _FakeStdioClient(params))
    monkeypatch.setattr(tools_mod, "ClientSession", _FakeSession)
    _FakeSession.last_call_kwargs = {}


@pytest.mark.asyncio
async def test_tool_passes_read_timeout_as_timedelta():
    tool = _make_langchain_tool("srv", _FakeMcpTool(), object(), tool_timeout_seconds=30.0)
    result = await tool.ainvoke({"q": "hello"})

    assert result == "tool output"
    kwargs = _FakeSession.last_call_kwargs
    assert kwargs["name"] == "search"
    assert kwargs["read_timeout_seconds"] == timedelta(seconds=30.0)


@pytest.mark.asyncio
async def test_tool_no_timeout_passes_none():
    tool = _make_langchain_tool("srv", _FakeMcpTool(), object(), tool_timeout_seconds=None)
    await tool.ainvoke({"q": "hello"})

    assert _FakeSession.last_call_kwargs["read_timeout_seconds"] is None


@pytest.mark.asyncio
async def test_tool_name_is_prefixed_with_server():
    tool = _make_langchain_tool("srv", _FakeMcpTool(name="fetch"), object())
    assert tool.name == "srv__fetch"
