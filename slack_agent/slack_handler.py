import asyncio
import logging
import re

from langchain_core.messages import HumanMessage
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from .config import AppConfig
from .state import AgentState
from .thread_history import fetch_new_replies

logger = logging.getLogger(__name__)

_ERROR_MESSAGE = "申し訳ありません。エラーが発生しました。しばらくしてから再度お試しください。"
_MENTION_PATTERN = re.compile(r"<@[^>]+>")


def create_app(config: AppConfig, compiled_graph, agent_config) -> AsyncApp:
    app = AsyncApp(token=config.slack.bot_token)

    # bot_user_id を memoize するためのクロージャ。auth_test の二重実行を避けるため
    # asyncio.Lock で初回取得を直列化する。
    bot_user_id_holder: dict[str, str] = {}
    bot_user_id_lock = asyncio.Lock()

    async def _get_bot_user_id(client) -> str:
        if "id" in bot_user_id_holder:
            return bot_user_id_holder["id"]
        async with bot_user_id_lock:
            if "id" in bot_user_id_holder:
                return bot_user_id_holder["id"]
            auth_result = await client.auth_test()
            bot_user_id_holder["id"] = auth_result.get("user_id", "")
            logger.info("Resolved bot_user_id=%s", bot_user_id_holder["id"])
            return bot_user_id_holder["id"]

    async def _handle_message(body: dict, say, client):
        event = body.get("event", {})
        user_id = event.get("user")
        text = event.get("text", "")
        channel = event.get("channel")
        event_ts = event.get("ts")
        thread_ts = event.get("thread_ts") or event_ts

        # Strip bot mention from text
        text = _MENTION_PATTERN.sub("", text).strip()

        # Check allowed users
        allowed = config.slack.allowed_user_ids
        if allowed and user_id not in allowed:
            logger.info("User %s not in allowed list, ignoring", user_id)
            return

        gconfig = {"configurable": {"thread_id": thread_ts}}

        # checkpoint から prior state を取得し、差分取得 or フォールバックを判定
        prior_values: dict = {}
        try:
            prior = await compiled_graph.aget_state(gconfig)
            prior_values = prior.values if prior else {}
        except Exception as exc:
            logger.warning("Failed to get prior state for thread=%s: %s", thread_ts, exc)

        messages_in_state = prior_values.get("messages", [])
        last_bot_response_ts = prior_values.get("last_bot_response_ts")

        # messages_in_state が空 = checkpoint 無し / 再起動後 -> 全件フォールバック
        # 非空 -> last_bot_response_ts 以降の差分のみ取得
        if not messages_in_state:
            since_ts = None
            mode = "fallback"
        else:
            since_ts = last_bot_response_ts
            mode = "diff"

        bot_user_id = await _get_bot_user_id(client)
        new_replies = await fetch_new_replies(
            client,
            channel=channel,
            thread_root_ts=thread_ts,
            since_ts=since_ts,
            current_ts=event_ts,
            bot_user_id=bot_user_id,
        )

        logger.info(
            "thread=%s mode=%s prior_messages=%d new_replies=%d since=%s event_ts=%s text=%r",
            thread_ts, mode, len(messages_in_state), len(new_replies), since_ts, event_ts, text[:60],
        )
        for i, m in enumerate(new_replies):
            logger.info("  new_replies[%d]: %r", i, str(m.content)[:80])

        # 進捗メッセージ: 1件のみ投稿し、推論ステップを追記しながら更新
        progress_ts: list[str] = []
        progress_lines: list[str] = []

        async def slack_notify(message: str):
            progress_lines.append(message)
            combined = "\n".join(progress_lines)
            if not progress_ts:
                result = await client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=combined,
                )
                progress_ts.append(result["ts"])
            else:
                await client.chat_update(
                    channel=channel,
                    ts=progress_ts[0],
                    text=combined,
                )

        current_human = HumanMessage(content=text)
        # last_bot_response_ts は initial_state に含めない (既存値を維持)
        initial_state: AgentState = {
            "messages": new_replies + [current_human],
            "compression_threshold": agent_config.compression_threshold_bytes,
            "cache_references": [],
            "pending_progress_message": None,
        }

        try:
            final_state = await compiled_graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": agent_config.recursion_limit,
                    "configurable": {
                        "slack_notify": slack_notify,
                        "thread_id": thread_ts,
                    },
                },
            )
            final_message = final_state["messages"][-1]
            answer = str(final_message.content)
        except Exception as exc:
            logger.exception("Agent failed: %s", exc)
            answer = _ERROR_MESSAGE

        # 最終回答は別メッセージとして投稿（進捗メッセージとは別スレッド返信）
        result = await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=answer,
        )

        # 次回の差分取得整合性のため、エラー応答時も last_bot_response_ts を更新する
        posted_ts = result.get("ts")
        if posted_ts:
            try:
                await compiled_graph.aupdate_state(gconfig, {"last_bot_response_ts": posted_ts})
            except Exception as exc:
                logger.warning(
                    "Failed to update last_bot_response_ts (thread=%s ts=%s): %s",
                    thread_ts, posted_ts, exc,
                )

    @app.event("message")
    async def handle_dm(body, say, client):
        event = body.get("event", {})
        # Only handle DMs (channel_type == "im") and non-bot messages
        if event.get("subtype") is not None:
            return
        if event.get("bot_id") is not None:
            return
        if event.get("channel_type") != "im":
            return
        await _handle_message(body, say, client)

    @app.event("app_mention")
    async def handle_mention(body, say, client):
        event = body.get("event", {})
        if event.get("subtype") is not None:
            return
        await _handle_message(body, say, client)

    return app


async def run_app(config: AppConfig, app: AsyncApp):
    handler = AsyncSocketModeHandler(app, config.slack.app_token)
    await handler.start_async()
