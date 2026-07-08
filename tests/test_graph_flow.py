"""build_graph で組んだ実グラフの遷移を統合的に検証する。

fake LLM / fake tool を差し替えて compile し、astream(stream_mode="updates")
でノードの遷移順を観測しつつ、ainvoke で最終 state を検証する。
LangGraph 固有のテスト API は不要で、compiled graph を実行して
ノード遷移と state を見るのが標準的なやり方。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from slack_agent.config import (
    AgentConfig,
    AppConfig,
    ModelConfig,
    RetryConfig,
    SlackConfig,
    StorageConfig,
)
from slack_agent.checkpointer import create_checkpointer
from slack_agent.graph import build_graph


# ---- fakes ---------------------------------------------------------------


class _ScriptedLLM:
    """ainvoke のたびに事前に並べた応答を順に返す fake LLM。

    bind_tools は self を返すだけ (tool バインドはテストに不要)。
    """

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


class _FakeTool:
    def __init__(self, result: str):
        self._result = result

    async def ainvoke(self, args):
        return self._result


def _config() -> AppConfig:
    return AppConfig(
        slack=SlackConfig(bot_token="b", app_token="a", allowed_user_ids=[]),
        standard_model=ModelConfig(model="x:y", options={}),
        retry=RetryConfig(max_attempts=1, backoff_base_seconds=0),
        agent=AgentConfig(recursion_limit=25, progress_mode="auto", mcp_tool_timeout_seconds=60),
        storage=StorageConfig(type="memory"),
    )


def _tool_call_ai(tool_name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id}])


def _initial_state(text: str) -> dict:
    return {
        "messages": [HumanMessage(content=text)],
    }


async def _collect_node_sequence(graph, state, config=None) -> list[str]:
    """astream(stream_mode="updates") から実行ノード名の系列を収集する。"""
    seq: list[str] = []
    async for update in graph.astream(state, stream_mode="updates", config=config):
        seq.extend(update.keys())
    return seq


# ---- tests ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_immediate_answer_ends_without_tools():
    """orchestrator が tool_calls 無しを返したら即 END。"""
    llm = _ScriptedLLM([AIMessage(content="direct answer")])
    graph = build_graph(_config(), llm, {})

    seq = await _collect_node_sequence(graph, _initial_state("hi"))
    assert seq == ["orchestrator"]

    final = await build_graph(
        _config(), _ScriptedLLM([AIMessage(content="direct answer")]), {},
    ).ainvoke(_initial_state("hi"))
    assert final["messages"][-1].content == "direct answer"


@pytest.mark.asyncio
async def test_tool_call_round_trip():
    """tool 実行 → orchestrator へ戻り END。"""
    responses = [
        _tool_call_ai("srv__search", {"q": "x"}, "tc1"),
        AIMessage(content="answer after tool"),
    ]
    tools = {"srv__search": _FakeTool("result")}
    graph = build_graph(_config(), _ScriptedLLM(responses), tools)

    seq = await _collect_node_sequence(graph, _initial_state("search x"))
    assert seq == ["orchestrator", "tool_executor", "orchestrator"]


@pytest.mark.asyncio
async def test_multi_tool_round_trips():
    """tool を 2 回呼んでから最終回答に到達する経路。"""
    responses = [
        _tool_call_ai("srv__search", {"q": "a"}, "tc1"),
        _tool_call_ai("srv__search", {"q": "b"}, "tc2"),
        AIMessage(content="done"),
    ]
    tools = {"srv__search": _FakeTool("small")}
    graph = build_graph(_config(), _ScriptedLLM(responses), tools)

    seq = await _collect_node_sequence(graph, _initial_state("multi"))
    assert seq == [
        "orchestrator", "tool_executor",
        "orchestrator", "tool_executor",
        "orchestrator",
    ]


@pytest.mark.asyncio
async def test_checkpointer_persists_history_across_invocations():
    """同じ thread_id で 2 回 invoke すると 1 回目の履歴が引き継がれる。"""
    checkpointer = create_checkpointer(StorageConfig(type="memory"))
    # 1 回目・2 回目とも tool 無しで即答する LLM
    llm = _ScriptedLLM([
        AIMessage(content="first answer"),
        AIMessage(content="second answer"),
    ])
    graph = build_graph(_config(), llm, {}, checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "thread-1"}}

    first = await graph.ainvoke(_initial_state("hello"), config=cfg)
    assert first["messages"][-1].content == "first answer"
    first_len = len(first["messages"])

    # 2 回目: HumanMessage を 1 件だけ足して同じ thread_id で invoke
    second = await graph.ainvoke(
        {"messages": [HumanMessage(content="follow up")]}, config=cfg
    )
    # checkpointer により 1 回目の履歴 + 追加分が積み上がっている
    assert len(second["messages"]) == first_len + 2  # follow up + second answer
    contents = [str(m.content) for m in second["messages"]]
    assert "hello" in contents
    assert "first answer" in contents
    assert "follow up" in contents
    assert second["messages"][-1].content == "second answer"


@pytest.mark.asyncio
async def test_checkpointer_isolates_distinct_threads():
    """異なる thread_id の state は混ざらない。"""
    checkpointer = create_checkpointer(StorageConfig(type="memory"))
    llm = _ScriptedLLM([
        AIMessage(content="answer A"),
        AIMessage(content="answer B"),
    ])
    graph = build_graph(_config(), llm, {}, checkpointer=checkpointer)

    a = await graph.ainvoke(_initial_state("thread A msg"), config={"configurable": {"thread_id": "A"}})
    b = await graph.ainvoke(_initial_state("thread B msg"), config={"configurable": {"thread_id": "B"}})

    a_contents = [str(m.content) for m in a["messages"]]
    b_contents = [str(m.content) for m in b["messages"]]
    assert "thread A msg" in a_contents and "thread B msg" not in a_contents
    assert "thread B msg" in b_contents and "thread A msg" not in b_contents


class _RecordingReporter:
    """progress_reporter のモック。update_task の呼び出しを記録する。"""

    def __init__(self):
        self.events: list[tuple] = []

    async def update_task(self, task_id, *, title=None, status="in_progress", output=None, details=None):
        self.events.append((task_id, title, status, details))

    async def finish(self):
        pass


@pytest.mark.asyncio
async def test_progress_reporter_receives_task_transitions():
    """configurable 経由で reporter が tool task の in_progress→complete を受け取る。"""
    responses = [
        _tool_call_ai("srv__search", {"q": "x"}, "tc1"),
        AIMessage(content="done"),
    ]
    tools = {"srv__search": _FakeTool("small")}
    graph = build_graph(_config(), _ScriptedLLM(responses), tools)

    reporter = _RecordingReporter()
    await graph.ainvoke(
        _initial_state("search x"),
        config={"configurable": {"progress_reporter": reporter}},
    )

    # tc1 が in_progress → complete と遷移する
    statuses = [(tid, st) for tid, _title, st, _details in reporter.events]
    assert ("tc1", "in_progress") in statuses
    assert ("tc1", "complete") in statuses
    # in_progress が complete より先
    assert statuses.index(("tc1", "in_progress")) < statuses.index(("tc1", "complete"))
    # 経緯テキストが無い場合、title はツール名にフォールバックする
    in_progress = [e for e in reporter.events if e[2] == "in_progress"][0]
    assert in_progress[1] == "srv__search"


@pytest.mark.asyncio
async def test_progress_reporter_reasoning_in_title_dump_in_details():
    """経緯は先頭タスクの title (見出し)、ツール dump は各タスクの details に出る。"""
    responses = [
        AIMessage(
            content="進捗を確認します",
            tool_calls=[
                {"name": "srv__search", "args": {"q": "a"}, "id": "tc1"},
                {"name": "srv__search", "args": {"q": "b"}, "id": "tc2"},
            ],
        ),
        AIMessage(content="done"),
    ]
    tools = {"srv__search": _FakeTool("small")}
    graph = build_graph(_config(), _ScriptedLLM(responses), tools)

    reporter = _RecordingReporter()
    await graph.ainvoke(
        _initial_state("search"),
        config={"configurable": {"progress_reporter": reporter}},
    )

    in_progress = {
        tid: (title, details)
        for tid, title, st, details in reporter.events
        if st == "in_progress"
    }
    # 先頭タスク (tc1) の title に経緯、後続 (tc2) はツール名
    assert in_progress["tc1"][0] == "進捗を確認します"
    assert in_progress["tc2"][0] == "srv__search"
    # ツール dump は各タスクの details に入る (引数を含む)
    assert in_progress["tc1"][1] == 'srv__search {"q": "a"}'
    assert in_progress["tc2"][1] == 'srv__search {"q": "b"}'
