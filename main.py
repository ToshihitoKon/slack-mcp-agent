import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from slack_agent.cache import InMemoryCacheStore
from slack_agent.checkpointer import create_checkpointer
from slack_agent.config import build_llm, load_config, _expand_recursive
from slack_agent.graph import build_graph
from slack_agent.slack_handler import create_app, run_app
from slack_agent.tools import load_mcp_tools, make_cache_fetcher_tool, _load_server_tools

# DEBUG_LLM=1 で LangChain の verbose/debug を有効にし、LLM への入出力を全部出す
if os.environ.get("DEBUG_LLM"):
    from langchain_core.globals import set_debug, set_verbose
    set_debug(True)
    set_verbose(True)
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


async def dry_run(mcp_config_path: str):
    raw = json.loads(Path(mcp_config_path).read_text())
    raw = _expand_recursive(raw)
    servers: dict = raw.get("mcpServers", {})

    ok = []
    failed = []
    for server_name, server_cfg in servers.items():
        print(f"[{server_name}] connecting... ", end="", flush=True)
        try:
            tools = await _load_server_tools(server_name, server_cfg)
            tool_names = ", ".join(t.name for t in tools)
            print(f"OK ({len(tools)} tools: {tool_names})")
            ok.append(server_name)
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed.append(server_name)

    print(f"\nResult: {len(ok)}/{len(ok) + len(failed)} servers OK")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(description="Slack MCP Agent")
    parser.add_argument("--mcp-check", action="store_true", help="Test MCP server connections and exit")
    parser.add_argument("--mcp-config", default="mcp_config.json", help="Path to MCP config file")
    parser.add_argument("--settings", default="settings.json", help="Path to settings file")
    args = parser.parse_args()

    if args.mcp_check:
        await dry_run(args.mcp_config)
        return

    config = load_config(args.settings)

    standard_llm = build_llm(config.standard_model)
    light_llm = build_llm(config.light_model)

    cache_store = InMemoryCacheStore()

    mcp_tools = await load_mcp_tools(args.mcp_config, cache_store)
    cache_fetcher = make_cache_fetcher_tool(cache_store)
    all_tools = mcp_tools + [cache_fetcher]

    # Bind tools to the standard LLM for orchestrator
    standard_llm = standard_llm.bind_tools(all_tools)

    tools_by_name = {t.name: t for t in all_tools}

    extra_prompt = ""
    prompt_path = Path("prompt.md")
    if prompt_path.exists():
        extra_prompt = prompt_path.read_text()
        logger.info("Loaded prompt.md as initial prompt")

    checkpointer = create_checkpointer(config.storage)
    logger.info("Checkpointer initialized: type=%s", config.storage.type)

    compiled_graph = build_graph(
        config=config,
        standard_llm=standard_llm,
        light_llm=light_llm,
        tools_by_name=tools_by_name,
        cache_store=cache_store,
        extra_prompt=extra_prompt,
        checkpointer=checkpointer,
    )

    app = create_app(config, compiled_graph, config.agent)

    logger.info("Starting Slack MCP Agent...")
    await run_app(config, app)


if __name__ == "__main__":
    asyncio.run(main())
