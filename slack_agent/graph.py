import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, StateGraph

from .cache import CacheStore
from .config import AppConfig
from .nodes import compressor_node, orchestrator_node, tool_executor_node
from .state import AgentState

logger = logging.getLogger(__name__)


def _should_compress(state: AgentState) -> str:
    threshold = state.get("compression_threshold", 10000)
    messages = state["messages"]
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if len(str(msg.content).encode()) > threshold:
                return "compressor"
            break
    return "orchestrator"


def _after_orchestrator(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_executor"
    return END


def build_graph(
    config: AppConfig,
    standard_llm: BaseChatModel,
    light_llm: BaseChatModel,
    tools_by_name: dict,
    cache_store: CacheStore,
    slack_notify_func: Callable | None = None,
):
    graph = StateGraph(AgentState)

    async def orchestrator(state: AgentState) -> dict:
        return await orchestrator_node(state, standard_llm, config.retry)

    async def tool_executor(state: AgentState) -> dict:
        return await tool_executor_node(state, tools_by_name, slack_notify_func, config.retry)

    async def compressor(state: AgentState) -> dict:
        return await compressor_node(
            state, light_llm, cache_store, config.cache.ttl_hours, config.retry
        )

    graph.add_node("orchestrator", orchestrator)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("compressor", compressor)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges("orchestrator", _after_orchestrator, {
        "tool_executor": "tool_executor",
        END: END,
    })
    graph.add_conditional_edges("tool_executor", _should_compress, {
        "compressor": "compressor",
        "orchestrator": "orchestrator",
    })
    graph.add_edge("compressor", "orchestrator")

    return graph.compile()
