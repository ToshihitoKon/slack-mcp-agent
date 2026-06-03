"""graph.py のルーティング関数 (_should_compress / _after_orchestrator) を検証する。"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from slack_agent.graph import _after_orchestrator, _should_compress


def _tool_msg(content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="tc1")


def test_should_compress_routes_to_compressor_when_over_threshold():
    state = {
        "messages": [_tool_msg("X" * 20000)],
        "compression_threshold": 10000,
    }
    assert _should_compress(state) == "compressor"


def test_should_compress_routes_to_orchestrator_when_under_threshold():
    state = {
        "messages": [_tool_msg("small")],
        "compression_threshold": 10000,
    }
    assert _should_compress(state) == "orchestrator"


def test_should_compress_inspects_only_last_tool_message():
    # 直近の ToolMessage が閾値以下なら、過去の大きい ToolMessage は無視される
    state = {
        "messages": [_tool_msg("X" * 20000), _tool_msg("small")],
        "compression_threshold": 10000,
    }
    assert _should_compress(state) == "orchestrator"


def test_should_compress_routes_to_orchestrator_without_tool_message():
    state = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="hello")],
        "compression_threshold": 10000,
    }
    assert _should_compress(state) == "orchestrator"


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
