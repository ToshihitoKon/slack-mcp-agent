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
STATUS_ERROR = "error"

# 応答生成全体を表す最上位タスク。ツール呼び出しタスクが全て complete に
# なった瞬間、これが無いと Plan Block 全体が complete 扱いになり見出しの
# 表示が完了扱いになってしまう。最初に追加するタスクとして起動時に
# in_progress にし、最終回答が確定したときだけ complete にする。
RESPONDING_TASK_ID = "__responding__"

# task_update chunk の title / output は 256 文字制限がある
_PLAN_TEXT_LIMIT = 256


def _truncate(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= _PLAN_TEXT_LIMIT:
        return text
    return text[: _PLAN_TEXT_LIMIT - 1] + "…"


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
        details: str | None = None,
    ) -> None:
        ...

    @abstractmethod
    async def finish(self) -> None:
        ...


class PlanBlockReporter(ProgressReporter):
    """Plan Block を chat.startStream / chat.appendStream で送る reporter。

    chat.startStream は recipient_team_id が必須 (無いと missing_recipient_team_id)。
    DM など 1:1 の文脈では recipient_user_id も渡す。
    """

    def __init__(
        self,
        client,
        channel: str,
        thread_ts: str,
        recipient_team_id: str | None = None,
        recipient_user_id: str | None = None,
    ):
        self._client = client
        self._channel = channel
        self._thread_ts = thread_ts
        self._recipient_team_id = recipient_team_id
        self._recipient_user_id = recipient_user_id
        self._stream_ts: str | None = None
        # task_id -> title を保持し、status 更新時に title を再送できるようにする
        self._titles: dict[str, str] = {}
        # details (ツール dump) を送信済みの task_id。Slack 側で追記される
        # ため、二重送信を防ぐ「送ったか」の記録として使う。
        self._details: dict[str, str] = {}

    async def start(self) -> None:
        kwargs: dict = {
            "channel": self._channel,
            "thread_ts": self._thread_ts,
            "task_display_mode": "plan",
        }
        if self._recipient_team_id is not None:
            kwargs["recipient_team_id"] = self._recipient_team_id
        if self._recipient_user_id is not None:
            kwargs["recipient_user_id"] = self._recipient_user_id
        logger.debug("chat.startStream request kwargs=%r", kwargs)
        result = await self._client.chat_startStream(**kwargs)
        logger.debug("chat.startStream response=%r", result)
        self._stream_ts = result["ts"]

    async def update_task(
        self,
        task_id: str,
        *,
        title: str | None = None,
        status: str = STATUS_IN_PROGRESS,
        output: str | None = None,
        details: str | None = None,
    ) -> None:
        if title is not None:
            self._titles[task_id] = title
        # chat.appendStream の task chunk スキーマ:
        #   type="task_update", id, title, status, (details/output/sources)
        # title/output/details は 256 文字制限。
        chunk: dict = {
            "type": "task_update",
            "id": task_id,
            "title": _truncate(self._titles.get(task_id, task_id)),
            "status": status,
        }
        # details は Slack 側で同一 task_id に追記 (append) される。同じ
        # status 遷移 (in_progress -> complete) で 2 回送ると dump が二重に
        # 表示されるため、各 task_id につき 1 度だけ送る。
        if details is not None and task_id not in self._details:
            self._details[task_id] = details
            chunk["details"] = _truncate(details)
        if output is not None:
            chunk["output"] = _truncate(output)
        logger.debug("chat.appendStream request chunk=%r", chunk)
        result = await self._client.chat_appendStream(
            channel=self._channel,
            ts=self._stream_ts,
            chunks=[chunk],
        )
        logger.debug("chat.appendStream response=%r", result)

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
            # ツール呼び出し前のエージェント思考はタスク行の前に出す。
            details = task.get("details")
            if details:
                lines.append(f"> {details}")
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
        details: str | None = None,
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
        if details is not None:
            task["details"] = details

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
    """初回の update_task まで実際の reporter 生成 (= Slack 投稿) を遅延する。"""

    def __init__(
        self,
        client,
        channel: str,
        thread_ts: str,
        mode: str = "auto",
        recipient_team_id: str | None = None,
        recipient_user_id: str | None = None,
    ):
        self._client = client
        self._channel = channel
        self._thread_ts = thread_ts
        self._mode = mode
        self._recipient_team_id = recipient_team_id
        self._recipient_user_id = recipient_user_id
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
        details: str | None = None,
    ) -> None:
        # 進捗表示はベストエフォート。Slack 側のスキーマ/権限エラーで
        # エージェント本体の処理を止めない。
        try:
            if self._inner is None:
                self._inner = await create_reporter(
                    self._client,
                    self._channel,
                    self._thread_ts,
                    self._mode,
                    recipient_team_id=self._recipient_team_id,
                    recipient_user_id=self._recipient_user_id,
                )
            await self._inner.update_task(
                task_id, title=title, status=status, output=output, details=details
            )
        except Exception as exc:
            logger.warning("Progress update failed (ignored): %s", exc)

    async def finish(self) -> None:
        if self._inner is not None:
            try:
                await self._inner.finish()
            except Exception as exc:
                logger.warning("Progress finish failed (ignored): %s", exc)


async def create_reporter(
    client,
    channel: str,
    thread_ts: str,
    mode: str = "auto",
    recipient_team_id: str | None = None,
    recipient_user_id: str | None = None,
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

    def _plan() -> PlanBlockReporter:
        return PlanBlockReporter(
            client, channel, thread_ts,
            recipient_team_id=recipient_team_id,
            recipient_user_id=recipient_user_id,
        )

    if mode == "plan":
        reporter = _plan()
        await reporter.start()
        return reporter

    # auto
    plan = _plan()
    try:
        await plan.start()
        return plan
    except Exception as exc:
        logger.info("Plan Block unavailable, falling back to text progress: %s", exc)
        text = TextProgressReporter(client, channel, thread_ts)
        await text.start()
        return text
