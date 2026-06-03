"""Slack への進捗表示を抽象化するモジュール (Issue #6)。

進捗を「タスクの集合とその状態遷移」として表現し、2 つの送信実装を提供する:

- PlanBlockReporter: Slack の Plan Block (chat.startStream / chat.appendStream)
  を使い、tool_call を 1 タスクとして構造化表示する。AI agent 機能が有効で
  assistant:write スコープを持つワークスペースでのみ動作する。
- TextProgressReporter: chat_postMessage / chat_update で 1 メッセージを
  更新し続ける従来のテキスト blockquote 表示。Plan Block 非対応環境向け。

create_reporter(...) は mode="auto" の場合に Plan Block を試み、開始に失敗
したら Text にフォールバックする。
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Slack Plan Block の task status 値
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETE = "complete"


class ProgressReporter(ABC):
    """進捗表示の抽象基底。

    使い方:
        await reporter.start()
        await reporter.update_task("task_1", title="...", status=STATUS_IN_PROGRESS)
        await reporter.update_task("task_1", status=STATUS_COMPLETE, output="...")
        await reporter.finish()
    """

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def update_task(
        self,
        task_id: str,
        *,
        title: str | None = None,
        status: str = STATUS_IN_PROGRESS,
        output: str | None = None,
    ) -> None:
        ...

    @abstractmethod
    async def finish(self) -> None:
        ...


class PlanBlockReporter(ProgressReporter):
    """Plan Block を chat.startStream / chat.appendStream で送る reporter。"""

    def __init__(self, client, channel: str, thread_ts: str):
        self._client = client
        self._channel = channel
        self._thread_ts = thread_ts
        self._stream_ts: str | None = None
        # task_id -> title を保持し、status 更新時に title を再送できるようにする
        self._titles: dict[str, str] = {}

    async def start(self) -> None:
        result = await self._client.chat_startStream(
            channel=self._channel,
            thread_ts=self._thread_ts,
            task_display_mode="plan",
        )
        self._stream_ts = result["ts"]

    async def update_task(
        self,
        task_id: str,
        *,
        title: str | None = None,
        status: str = STATUS_IN_PROGRESS,
        output: str | None = None,
    ) -> None:
        if title is not None:
            self._titles[task_id] = title
        chunk = {
            "type": "task",
            "id": task_id,
            "text": self._titles.get(task_id, task_id),
            "status": status,
        }
        if output is not None:
            chunk["output"] = output
        await self._client.chat_appendStream(
            channel=self._channel,
            ts=self._stream_ts,
            chunks=[chunk],
        )

    async def finish(self) -> None:
        if self._stream_ts is None:
            return
        await self._client.chat_stopStream(
            channel=self._channel,
            ts=self._stream_ts,
        )


class TextProgressReporter(ProgressReporter):
    """テキスト blockquote を 1 メッセージで更新し続ける従来表示。

    タスクごとに 1 行を持ち、status 変化のたびにメッセージ全体を再構築して
    chat_update する。
    """

    _STATUS_ICON = {
        STATUS_PENDING: "◻️",
        STATUS_IN_PROGRESS: "⏳",
        STATUS_COMPLETE: "✅",
    }

    def __init__(self, client, channel: str, thread_ts: str):
        self._client = client
        self._channel = channel
        self._thread_ts = thread_ts
        self._message_ts: str | None = None
        # 表示順を保つため task_id のリストと内容を別管理
        self._order: list[str] = []
        self._tasks: dict[str, dict] = {}

    async def start(self) -> None:
        # 最初の update_task まで実際の投稿は遅延させる (空メッセージを作らない)
        return None

    def _render(self) -> str:
        lines = []
        for task_id in self._order:
            task = self._tasks[task_id]
            icon = self._STATUS_ICON.get(task["status"], "•")
            text = task.get("title") or task_id
            lines.append(f"> {icon} _{text}_")
            output = task.get("output")
            if output:
                lines.append(f"> {output}")
        return "\n".join(lines)

    async def update_task(
        self,
        task_id: str,
        *,
        title: str | None = None,
        status: str = STATUS_IN_PROGRESS,
        output: str | None = None,
    ) -> None:
        if task_id not in self._tasks:
            self._order.append(task_id)
            self._tasks[task_id] = {}
        task = self._tasks[task_id]
        if title is not None:
            task["title"] = title
        task["status"] = status
        if output is not None:
            task["output"] = output

        combined = self._render()
        if self._message_ts is None:
            result = await self._client.chat_postMessage(
                channel=self._channel,
                thread_ts=self._thread_ts,
                text=combined,
            )
            self._message_ts = result["ts"]
        else:
            await self._client.chat_update(
                channel=self._channel,
                ts=self._message_ts,
                text=combined,
            )

    async def finish(self) -> None:
        # テキストモードでは特に終了処理は不要
        return None


class LazyReporter(ProgressReporter):
    """初回の update_task まで実際の reporter 生成 (= Slack 投稿) を遅延する。

    tool_call が無く即答するケースで空のストリーム/メッセージを作らないため。
    """

    def __init__(self, client, channel: str, thread_ts: str, mode: str = "auto"):
        self._client = client
        self._channel = channel
        self._thread_ts = thread_ts
        self._mode = mode
        self._inner: ProgressReporter | None = None

    async def start(self) -> None:
        return None

    async def update_task(
        self,
        task_id: str,
        *,
        title: str | None = None,
        status: str = STATUS_IN_PROGRESS,
        output: str | None = None,
    ) -> None:
        if self._inner is None:
            self._inner = await create_reporter(
                self._client, self._channel, self._thread_ts, self._mode
            )
        await self._inner.update_task(
            task_id, title=title, status=status, output=output
        )

    async def finish(self) -> None:
        if self._inner is not None:
            await self._inner.finish()


async def create_reporter(
    client,
    channel: str,
    thread_ts: str,
    mode: str = "auto",
) -> ProgressReporter:
    """mode に応じた reporter を生成する。

    - "plan": PlanBlockReporter を使う (失敗してもフォールバックしない)
    - "text": TextProgressReporter を使う
    - "auto": Plan Block を試し、start に失敗したら Text にフォールバック
    """
    if mode == "text":
        reporter: ProgressReporter = TextProgressReporter(client, channel, thread_ts)
        await reporter.start()
        return reporter

    if mode == "plan":
        reporter = PlanBlockReporter(client, channel, thread_ts)
        await reporter.start()
        return reporter

    # auto
    plan = PlanBlockReporter(client, channel, thread_ts)
    try:
        await plan.start()
        return plan
    except Exception as exc:
        logger.info("Plan Block unavailable, falling back to text progress: %s", exc)
        text = TextProgressReporter(client, channel, thread_ts)
        await text.start()
        return text
