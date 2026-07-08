import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from .config import AppConfig
from .nodes import orchestrator_node, tool_executor_node
from .state import AgentState

logger = logging.getLogger(__name__)


def _after_orchestrator(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_executor"
    return END


def build_graph(
    config: AppConfig,
    standard_llm: BaseChatModel,
    tools_by_name: dict,
    extra_prompt: str = "",
    checkpointer: BaseCheckpointSaver | None = None,
):
    graph = StateGraph(AgentState)

    async def orchestrator(state: AgentState) -> dict:
        return await orchestrator_node(state, standard_llm, config.retry, extra_prompt)

    async def tool_executor(state: AgentState) -> dict:
        from langgraph.config import get_config
        rconfig = get_config()
        progress_reporter = rconfig.get("configurable", {}).get("progress_reporter")
        return await tool_executor_node(state, tools_by_name, progress_reporter, config.retry)

    graph.add_node("orchestrator", orchestrator)
    graph.add_node("tool_executor", tool_executor)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges("orchestrator", _after_orchestrator, {
        "tool_executor": "tool_executor",
        END: END,
    })
    graph.add_edge("tool_executor", "orchestrator")

    return graph.compile(checkpointer=checkpointer)
