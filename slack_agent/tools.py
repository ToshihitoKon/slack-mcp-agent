import json
import logging
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
) -> list[BaseTool]:
    """Load tools from all configured MCP servers.

    キャッシュ保存はツール実行レイヤー (tool_executor_node) と
    compressor_node に集約しているため、ここでは cache_store を扱わない。

    tool_timeout_seconds: ツール呼び出しの応答待ちタイムアウト秒。None なら無制限。
    """
    raw = json.loads(Path(mcp_config_path).read_text())
    raw = _expand_env_vars_in_config(raw)
    servers: dict = raw.get("mcpServers", {})

    all_tools: list[BaseTool] = []
    for server_name, server_cfg in servers.items():
        tools = await _load_server_tools(server_name, server_cfg, tool_timeout_seconds)
        all_tools.extend(tools)

    return all_tools


async def _load_server_tools(
    server_name: str,
    server_cfg: dict,
    tool_timeout_seconds: float | None = None,
) -> list[BaseTool]:
    """Connect to a single stdio MCP server and return its tools as LangChain tools."""
    command = server_cfg["command"]
    args = server_cfg.get("args", [])
    env = server_cfg.get("env", {})

    params = StdioServerParameters(command=command, args=args, env=env or None)

    tools: list[BaseTool] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            for mcp_tool in result.tools:
                lc_tool = _make_langchain_tool(
                    server_name, mcp_tool, params, tool_timeout_seconds
                )
                tools.append(lc_tool)
    return tools


def _make_langchain_tool(
    server_name: str,
    mcp_tool,
    params: StdioServerParameters,
    tool_timeout_seconds: float | None = None,
) -> BaseTool:
    """Wrap a single MCP tool as a LangChain BaseTool."""
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
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
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
