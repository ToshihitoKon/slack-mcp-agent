"""graph.py のルーティング関数 (_after_orchestrator) を検証する。"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from slack_agent.graph import _after_orchestrator


def test_after_orchestrator_routes_to_tool_executor_with_tool_calls():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "t", "args": {}, "id": "tc1"}],
    )
    state = {"messages": [HumanMessage(content="hi"), msg]}
    assert _after_orchestrator(state) == "tool_executor"


def test_after_orchestrator_ends_without_tool_calls():
    msg = AIMessage(content="final answer")
    state = {"messages": [HumanMessage(content="hi"), msg]}
    assert _after_orchestrator(state) == END
