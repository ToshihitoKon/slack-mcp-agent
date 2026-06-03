"""進捗 reporter (Plan Block / テキスト / 遅延 / フォールバック) を検証する (Issue #6)。"""

import pytest

from slack_agent.progress import (
    STATUS_COMPLETE,
    STATUS_IN_PROGRESS,
    LazyReporter,
    PlanBlockReporter,
    TextProgressReporter,
    create_reporter,
)


class _RecordingClient:
    """Slack client のモック。呼び出しを記録する。"""

    def __init__(self, start_stream_error: Exception | None = None):
        self._start_stream_error = start_stream_error
        self.calls: list[tuple] = []

    async def chat_startStream(self, **kwargs):
        self.calls.append(("startStream", kwargs))
        if self._start_stream_error is not None:
            raise self._start_stream_error
        return {"ts": "stream-1"}

    async def chat_appendStream(self, **kwargs):
        self.calls.append(("appendStream", kwargs))
        return {"ok": True}

    async def chat_stopStream(self, **kwargs):
        self.calls.append(("stopStream", kwargs))
        return {"ok": True}

    async def chat_postMessage(self, **kwargs):
        self.calls.append(("postMessage", kwargs))
        return {"ts": "msg-1"}

    async def chat_update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return {"ok": True}

    def kinds(self) -> list[str]:
        return [c[0] for c in self.calls]


# ---- PlanBlockReporter ---------------------------------------------------


@pytest.mark.asyncio
async def test_plan_block_starts_stream_with_plan_mode():
    client = _RecordingClient()
    reporter = PlanBlockReporter(client, "C1", "100")
    await reporter.start()

    kind, kwargs = client.calls[0]
    assert kind == "startStream"
    assert kwargs["channel"] == "C1"
    assert kwargs["thread_ts"] == "100"
    assert kwargs["task_display_mode"] == "plan"
    # 未指定なら recipient_* は送らない
    assert "recipient_team_id" not in kwargs
    assert "recipient_user_id" not in kwargs


@pytest.mark.asyncio
async def test_plan_block_passes_recipient_ids_when_given():
    """recipient_team_id は startStream に必須なので渡されたら送る。"""
    client = _RecordingClient()
    reporter = PlanBlockReporter(
        client, "C1", "100",
        recipient_team_id="T123", recipient_user_id="U456",
    )
    await reporter.start()

    _, kwargs = client.calls[0]
    assert kwargs["recipient_team_id"] == "T123"
    assert kwargs["recipient_user_id"] == "U456"


@pytest.mark.asyncio
async def test_create_reporter_auto_forwards_recipient_ids():
    """auto モードでも recipient_team_id / recipient_user_id が startStream に伝播する。

    チャンネルへのストリーミングでは両方が必須。
    """
    client = _RecordingClient()
    await create_reporter(
        client, "C1", "100", mode="auto",
        recipient_team_id="T999", recipient_user_id="U999",
    )

    start_calls = [kwargs for kind, kwargs in client.calls if kind == "startStream"]
    assert start_calls
    assert start_calls[0]["recipient_team_id"] == "T999"
    assert start_calls[0]["recipient_user_id"] == "U999"


@pytest.mark.asyncio
async def test_plan_block_appends_task_chunks_and_status_transition():
    client = _RecordingClient()
    reporter = PlanBlockReporter(client, "C1", "100")
    await reporter.start()

    await reporter.update_task("t1", title="Search workspace", status=STATUS_IN_PROGRESS)
    await reporter.update_task("t1", status=STATUS_COMPLETE)

    appends = [kwargs for kind, kwargs in client.calls if kind == "appendStream"]
    assert len(appends) == 2

    chunk1 = appends[0]["chunks"][0]
    assert chunk1 == {
        "type": "task",
        "id": "t1",
        "text": "Search workspace",
        "status": STATUS_IN_PROGRESS,
    }
    # status 更新時も title は保持される
    chunk2 = appends[1]["chunks"][0]
    assert chunk2["text"] == "Search workspace"
    assert chunk2["status"] == STATUS_COMPLETE


@pytest.mark.asyncio
async def test_plan_block_includes_output_when_given():
    client = _RecordingClient()
    reporter = PlanBlockReporter(client, "C1", "100")
    await reporter.start()
    await reporter.update_task("t1", title="x", status=STATUS_COMPLETE, output="done")

    chunk = client.calls[-1][1]["chunks"][0]
    assert chunk["output"] == "done"


@pytest.mark.asyncio
async def test_plan_block_finish_stops_stream():
    client = _RecordingClient()
    reporter = PlanBlockReporter(client, "C1", "100")
    await reporter.start()
    await reporter.finish()
    assert client.kinds()[-1] == "stopStream"


# ---- TextProgressReporter ------------------------------------------------


@pytest.mark.asyncio
async def test_text_reporter_posts_then_updates():
    client = _RecordingClient()
    reporter = TextProgressReporter(client, "C1", "100")
    await reporter.start()
    # start では投稿しない (遅延)
    assert client.calls == []

    await reporter.update_task("t1", title="first", status=STATUS_IN_PROGRESS)
    await reporter.update_task("t1", status=STATUS_COMPLETE)

    assert client.kinds() == ["postMessage", "update"]
    # 最初の投稿は postMessage、以降は同じ ts を update
    assert client.calls[1][1]["ts"] == "msg-1"


@pytest.mark.asyncio
async def test_text_reporter_renders_multiple_tasks_in_order():
    client = _RecordingClient()
    reporter = TextProgressReporter(client, "C1", "100")
    await reporter.start()
    await reporter.update_task("t1", title="alpha", status=STATUS_COMPLETE)
    await reporter.update_task("t2", title="beta", status=STATUS_IN_PROGRESS)

    last_text = client.calls[-1][1]["text"]
    assert "alpha" in last_text
    assert "beta" in last_text
    assert last_text.index("alpha") < last_text.index("beta")


# ---- create_reporter (fallback) -----------------------------------------


@pytest.mark.asyncio
async def test_create_reporter_auto_uses_plan_when_available():
    client = _RecordingClient()
    reporter = await create_reporter(client, "C1", "100", mode="auto")
    assert isinstance(reporter, PlanBlockReporter)


@pytest.mark.asyncio
async def test_create_reporter_auto_falls_back_to_text_on_error():
    client = _RecordingClient(start_stream_error=RuntimeError("not enabled"))
    reporter = await create_reporter(client, "C1", "100", mode="auto")
    assert isinstance(reporter, TextProgressReporter)


@pytest.mark.asyncio
async def test_create_reporter_text_mode_forces_text():
    client = _RecordingClient()
    reporter = await create_reporter(client, "C1", "100", mode="text")
    assert isinstance(reporter, TextProgressReporter)


# ---- LazyReporter --------------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_reporter_defers_creation_until_first_task():
    client = _RecordingClient()
    reporter = LazyReporter(client, "C1", "100", mode="auto")
    await reporter.start()
    # まだ何も投稿していない
    assert client.calls == []

    await reporter.update_task("t1", title="x", status=STATUS_IN_PROGRESS)
    # 初回 update で startStream が走る
    assert client.kinds()[0] == "startStream"


@pytest.mark.asyncio
async def test_lazy_reporter_forwards_recipient_ids_to_plan_block():
    """LazyReporter が recipient_team_id / recipient_user_id を startStream まで伝播する。"""
    client = _RecordingClient()
    reporter = LazyReporter(
        client, "C1", "100", mode="auto",
        recipient_team_id="T1", recipient_user_id="U1",
    )
    await reporter.start()
    await reporter.update_task("t1", title="x", status=STATUS_IN_PROGRESS)

    start_calls = [kwargs for kind, kwargs in client.calls if kind == "startStream"]
    assert start_calls
    assert start_calls[0]["recipient_team_id"] == "T1"
    assert start_calls[0]["recipient_user_id"] == "U1"


@pytest.mark.asyncio
async def test_lazy_reporter_finish_without_tasks_is_noop():
    client = _RecordingClient()
    reporter = LazyReporter(client, "C1", "100", mode="auto")
    await reporter.start()
    await reporter.finish()
    # タスクが一度も無ければ Slack 呼び出しは発生しない
    assert client.calls == []
