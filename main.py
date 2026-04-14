import asyncio
import logging

from slack_agent.cache import InMemoryCacheStore
from slack_agent.config import build_llm, load_config
from slack_agent.graph import build_graph
from slack_agent.slack_handler import create_app, run_app
from slack_agent.tools import load_mcp_tools, make_cache_fetcher_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    config = load_config("settings.json")

    standard_llm = build_llm(config.standard_model)
    light_llm = build_llm(config.light_model)

    cache_store = InMemoryCacheStore()

    mcp_tools = await load_mcp_tools("mcp_config.json", cache_store)
    cache_fetcher = make_cache_fetcher_tool(cache_store)
    all_tools = mcp_tools + [cache_fetcher]

    # Bind tools to the standard LLM for orchestrator
    standard_llm = standard_llm.bind_tools(all_tools)

    tools_by_name = {t.name: t for t in all_tools}

    compiled_graph = build_graph(
        config=config,
        standard_llm=standard_llm,
        light_llm=light_llm,
        tools_by_name=tools_by_name,
        cache_store=cache_store,
    )

    app = create_app(config, compiled_graph, config.agent)

    logger.info("Starting Slack MCP Agent...")
    await run_app(config, app)


if __name__ == "__main__":
    asyncio.run(main())
