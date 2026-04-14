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

    async def _handle_message(body: dict, say, client):
        event = body.get("event", {})
        user_id = event.get("user")
        text = event.get("text", "")
        channel = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")

        # Strip bot mention from text
        import re
        text = re.sub(r"<@[^>]+>", "", text).strip()

        # Check allowed users
        allowed = config.slack.allowed_user_ids
        if allowed and user_id not in allowed:
            logger.info("User %s not in allowed list, ignoring", user_id)
            return

        # Progress message state: create once, update in-place
        progress_ts: list[str] = []

        async def slack_notify(message: str):
            if not progress_ts:
                result = await client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=message,
                )
                progress_ts.append(result["ts"])
            else:
                await client.chat_update(
                    channel=channel,
                    ts=progress_ts[0],
                    text=message,
                )

        initial_state: AgentState = {
            "messages": [HumanMessage(content=text)],
            "compression_threshold": agent_config.compression_threshold_bytes,
            "cache_references": [],
            "pending_progress_message": None,
        }

        try:
            final_state = await compiled_graph.ainvoke(
                initial_state,
                config={"recursion_limit": agent_config.recursion_limit},
            )
            final_message = final_state["messages"][-1]
            answer = str(final_message.content)
        except Exception as exc:
            logger.exception("Agent failed: %s", exc)
            answer = _ERROR_MESSAGE

        await say(text=answer, thread_ts=thread_ts)

    @app.event("message")
    async def handle_dm(body, say, client):
        event = body.get("event", {})
        # Only handle DMs (channel_type == "im") and non-bot messages
        if event.get("subtype") is not None:
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
