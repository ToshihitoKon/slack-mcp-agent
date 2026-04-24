"""Slack スレッドの差分取得モジュール。

LangGraph checkpointer に保存された state と組み合わせて、
"最後の bot 発言以降の新規メッセージだけを取り込む" 差分取得を提供する。
checkpoint が無い (since_ts is None) 場合は全件取得にフォールバックする。

bot 自身の過去発言は AIMessage ではなく HumanMessage(content="[assistant said]: ...")
として取り込む。これは tool_calls の整合性エラーを避けつつ、
LLM に対して role 情報をテキストタグで伝えるための妥協策。
"""

import logging
import re

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

_MENTION_PATTERN = re.compile(r"<@[^>]+>")


async def fetch_new_replies(
    client,
    channel: str,
    thread_root_ts: str,
    since_ts: str | None,
    current_ts: str,
    bot_user_id: str,
) -> list[HumanMessage]:
    """スレッドから「最後の bot 発言より後」の新規メッセージを取得する。

    Args:
        client: Slack AsyncWebClient
        channel: チャンネル ID
        thread_root_ts: スレッドのルート ts
        since_ts: 直近の bot 発言 ts。None なら全件取得 (フォールバック)
        current_ts: 今回トリガーになった発言の ts (重複投入防止のためスキップ)
        bot_user_id: 自分の bot user id (assistant 判定用)

    Returns:
        HumanMessage のリスト。assistant role のメッセージは
        "[assistant said]: ..." の prefix 付きで HumanMessage として返す。
        取得失敗時は空リスト。
    """
    kwargs = {"channel": channel, "ts": thread_root_ts}
    if since_ts is not None:
        kwargs["oldest"] = since_ts
        kwargs["inclusive"] = False

    try:
        result = await client.conversations_replies(**kwargs)
    except Exception as exc:
        logger.warning(
            "Failed to fetch thread replies (channel=%s thread=%s since=%s): %s",
            channel, thread_root_ts, since_ts, exc,
        )
        return []

    raw_msgs = result.get("messages", [])
    logger.debug(
        "conversations_replies returned %d messages (since=%s current_ts=%s)",
        len(raw_msgs), since_ts, current_ts,
    )
    for m in raw_msgs:
        logger.debug(
            "  reply ts=%s user=%s bot_id=%s subtype=%s text=%r",
            m.get("ts"), m.get("user"), m.get("bot_id"),
            m.get("subtype"), (m.get("text") or "")[:80],
        )

    messages: list[HumanMessage] = []
    for msg in raw_msgs:
        if msg.get("subtype") is not None:
            continue
        if msg.get("ts") == current_ts:
            continue
        # Slack の conversations.replies は oldest を指定しても thread の親メッセージを
        # 常に先頭に含める仕様。差分取得 (since_ts is not None) では既に過去の messages に
        # 含まれているため除外する。フォールバック (since_ts is None) では取り込む。
        if since_ts is not None and msg.get("ts") == thread_root_ts:
            continue
        text = _MENTION_PATTERN.sub("", msg.get("text", "")).strip()
        if not text:
            continue
        is_assistant = msg.get("user") == bot_user_id or msg.get("bot_id")
        if is_assistant:
            messages.append(HumanMessage(content=f"[assistant said]: {text}"))
        else:
            messages.append(HumanMessage(content=text))

    return messages
