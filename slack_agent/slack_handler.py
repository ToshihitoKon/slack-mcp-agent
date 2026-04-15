import logging

from langchain_core.messages import HumanMessage
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from .config import AppConfig
from .state import AgentState

logger = logging.getLogger(__name__)

_ERROR_MESSAGE = "申し訳ありません。エラーが発生しました。しばらくしてから再度お試しください。"


def create_app(config: AppConfig, compiled_graph, agent_config) -> AsyncApp:
    app = AsyncApp(token=config.slack.bot_token)

    async def _fetch_thread_context(client, channel: str, thread_ts: str, bot_user_id: str, current_ts: str) -> str | None:
        """スレッドの過去メッセージをシステムプロンプト用のテキストとして返す"""
        try:
            result = await client.conversations_replies(
                channel=channel,
                ts=thread_ts,
            )
        except Exception as exc:
            logger.warning("Failed to fetch thread replies: %s", exc)
            return None

        import re
        lines = []
        for msg in result.get("messages", []):
            if msg.get("subtype") is not None:
                continue
            if msg.get("ts") == current_ts:
                continue
            msg_text = re.sub(r"<@[^>]+>", "", msg.get("text", "")).strip()
            if not msg_text:
                continue
            if msg.get("user") == bot_user_id or msg.get("bot_id"):
                role = "assistant"
            else:
                role = "user"
            lines.append(f"[{role}]: {msg_text}")

        if not lines:
            return None
        return "Previous thread context:\n" + "\n".join(lines)

    async def _handle_message(body: dict, say, client):
        event = body.get("event", {})
        user_id = event.get("user")
        text = event.get("text", "")
        channel = event.get("channel")
        event_ts = event.get("ts")
        thread_ts = event.get("thread_ts") or event_ts

        # Strip bot mention from text
        import re
        text = re.sub(r"<@[^>]+>", "", text).strip()

        # Check allowed users
        allowed = config.slack.allowed_user_ids
        if allowed and user_id not in allowed:
            logger.info("User %s not in allowed list, ignoring", user_id)
            return

        # スレッドの過去メッセージを取得（スレッド内の場合）
        thread_context: str | None = None
        if event.get("thread_ts"):
            auth_result = await client.auth_test()
            bot_user_id = auth_result.get("user_id", "")
            thread_context = await _fetch_thread_context(client, channel, thread_ts, bot_user_id, event_ts)

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

        human_message = HumanMessage(
            content=f"{thread_context}\n\n{text}" if thread_context else text
        )
        initial_state: AgentState = {
            "messages": [human_message],
            "compression_threshold": agent_config.compression_threshold_bytes,
            "cache_references": [],
            "pending_progress_message": None,
        }

        try:
            final_state = await compiled_graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": agent_config.recursion_limit,
                    "configurable": {"slack_notify": slack_notify},
                },
            )
            final_message = final_state["messages"][-1]
            answer = str(final_message.content)
        except Exception as exc:
            logger.exception("Agent failed: %s", exc)
            answer = _ERROR_MESSAGE

        # 最終回答は別メッセージとして投稿（進捗メッセージとは別スレッド返信）
        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=answer,
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
