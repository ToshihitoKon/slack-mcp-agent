import json
import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .cache import CacheStore

logger = logging.getLogger(__name__)


def _expand_env_vars_in_config(mcp_config: dict) -> dict:
    """Expand env vars in mcp_config (reuse config module logic if needed)."""
    from .config import _expand_recursive
    return _expand_recursive(mcp_config)


async def load_mcp_tools(
    mcp_config_path: str = "mcp_config.json",
    tool_timeout_seconds: float | None = None,
    exit_stack: AsyncExitStack | None = None,
) -> list[BaseTool]:
    """Load tools from all configured MCP servers.

    各 MCP サーバーへの接続は起動時に 1 回だけ確立し、その ClientSession を
    保持して使い回す。ツール呼び出しごとにサーバーを起動し直すと再起動レースで
    タイムアウトするため (Issue #6 調査)。

    exit_stack: セッションのライフサイクルを管理する AsyncExitStack。
        呼び出し元 (main) がアプリ終了まで開いたまま保持する。None の場合は
        この関数内で開くが、その場合セッションはすぐ閉じられるため通常は渡すこと。

    tool_timeout_seconds: ツール呼び出しの応答待ちタイムアウト秒。None なら無制限。

    キャッシュ保存はツール実行レイヤー (tool_executor_node) と
    compressor_node に集約しているため、ここでは cache_store を扱わない。
    """
    raw = json.loads(Path(mcp_config_path).read_text())
    raw = _expand_env_vars_in_config(raw)
    servers: dict = raw.get("mcpServers", {})

    if exit_stack is None:
        exit_stack = AsyncExitStack()
        await exit_stack.__aenter__()

    all_tools: list[BaseTool] = []
    for server_name, server_cfg in servers.items():
        tools = await _connect_server(
            server_name, server_cfg, tool_timeout_seconds, exit_stack
        )
        all_tools.extend(tools)

    return all_tools


async def _connect_server(
    server_name: str,
    server_cfg: dict,
    tool_timeout_seconds: float | None,
    exit_stack: AsyncExitStack,
) -> list[BaseTool]:
    """stdio MCP サーバーに接続し、保持したセッションを使う LangChain ツールを返す。

    stdio_client と ClientSession を exit_stack に登録し、アプリ終了まで
    セッションを生かしておく。"""
    command = server_cfg["command"]
    args = server_cfg.get("args", [])
    env = server_cfg.get("env", {})

    params = StdioServerParameters(command=command, args=args, env=env or None)

    read, write = await exit_stack.enter_async_context(stdio_client(params))
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()

    result = await session.list_tools()
    tools: list[BaseTool] = []
    for mcp_tool in result.tools:
        lc_tool = _make_langchain_tool(
            server_name, mcp_tool, session, tool_timeout_seconds
        )
        tools.append(lc_tool)
    logger.info("Connected to MCP server '%s' (%d tools)", server_name, len(tools))
    return tools


def _make_langchain_tool(
    server_name: str,
    mcp_tool,
    session: ClientSession,
    tool_timeout_seconds: float | None = None,
) -> BaseTool:
    """Wrap a single MCP tool as a LangChain BaseTool backed by a shared session."""
    tool_name = f"{server_name}__{mcp_tool.name}"
    tool_description = mcp_tool.description or ""
    input_schema = mcp_tool.inputSchema or {}
    # 応答が返らない MCP サーバーで無限ハングしないよう read_timeout を渡す。
    read_timeout = (
        timedelta(seconds=tool_timeout_seconds)
        if tool_timeout_seconds is not None
        else None
    )

    async def _arun(**kwargs: Any) -> str:
        # 保持済みセッションを使い回す。毎回サーバーを起動し直さない。
        # call_tool は直列化しない: ClientSession は request_id ごとに応答 stream を
        # 多重化するため並行呼び出しに対応している。直列化すると 1 本のハング
        # (esa の重いクエリ等) が後続の全ツールを巻き添えにする (連鎖タイムアウト)。
        result = await session.call_tool(
            mcp_tool.name,
            arguments=kwargs,
            read_timeout_seconds=read_timeout,
        )
        return str(result.content)

    return StructuredTool(
        name=tool_name,
        description=tool_description,
        args_schema=input_schema,
        coroutine=_arun,
        func=None,
    )


def make_cache_fetcher_tool(cache_store: CacheStore) -> BaseTool:
    """Return the built-in cache_fetcher tool."""

    class _CacheFetcher(BaseTool):
        name: str = "cache_fetcher"
        description: str = (
            "Fetch a previously cached tool result by cache_key. "
            "Returns the raw result if available, otherwise returns a cache miss message."
        )

        async def _arun(self, cache_key: str) -> str:
            entry = cache_store.get(cache_key)
            if entry is None:
                return f"Cache miss or expired for key: {cache_key}"
            return entry.raw_result

        def _run(self, cache_key: str) -> str:
            raise NotImplementedError("Use async")

    return _CacheFetcher()
