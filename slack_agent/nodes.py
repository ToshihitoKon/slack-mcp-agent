import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from .cache import CacheEntry, CacheStore
from .config import AppConfig
from .retry import retry_async
from .state import AgentState, CacheReference, CompressorResult

logger = logging.getLogger(__name__)

_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a helpful assistant with access to various tools.
Answer the user's questions accurately. Use tools when needed.
"""

_COMPRESSOR_SYSTEM_PROMPT = """\
You are a summarization assistant.
Given a tool result and the user's query, produce:
1. focused_summary: a concise summary relevant to the user's query
2. content_index: a structural index of the full result for future reference

Respond ONLY with valid JSON matching:
{"focused_summary": "...", "content_index": "..."}
"""


def _build_orchestrator_system(cache_references: list[CacheReference], extra_prompt: str = "") -> str:
    prompt = (extra_prompt.strip() + "\n\n" + _ORCHESTRATOR_SYSTEM_PROMPT) if extra_prompt.strip() else _ORCHESTRATOR_SYSTEM_PROMPT
    if cache_references:
        refs_text = "\n".join(
            f"- cache_key={r['cache_key']} tool={r['tool_name']} index={r['content_index']}"
            for r in cache_references
        )
        prompt += f"\n\nAvailable cached results (use cache_fetcher tool to retrieve):\n{refs_text}"
    return prompt


def _build_pending_message(response: AIMessage) -> str | None:
    if not response.tool_calls:
        return None
    lines = []
    # AIMessageのテキスト部分（思考・説明）があれば先に追加
    content = response.content
    def _quote(text: str) -> str:
        return "\n".join(f"> _{line}_" for line in text.strip().splitlines())

    if isinstance(content, str) and content.strip():
        lines.append(_quote(content))
    elif isinstance(content, list):
        # content blocksの場合（anthropicなど）
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                lines.append(_quote(block["text"]))
    # tool calls
    for tc in response.tool_calls:
        args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
        lines.append(f"> _{tc['name']} {args_str}_")
    return "\n".join(lines)


async def orchestrator_node(
    state: AgentState,
    standard_llm: BaseChatModel,
    retry_config,
    extra_prompt: str = "",
) -> dict:
    system_msg = SystemMessage(content=_build_orchestrator_system(state.get("cache_references", []), extra_prompt))
    messages = [system_msg] + list(state["messages"])

    logger.info("orchestrator invoke: %d messages", len(messages))
    for i, m in enumerate(messages):
        tc = getattr(m, "tool_calls", None)
        tcid = getattr(m, "tool_call_id", None)
        logger.info(
            "  [%d] %s tool_calls=%s tool_call_id=%s content=%r",
            i, type(m).__name__,
            [c.get("name") for c in tc] if tc else None,
            tcid,
            (str(m.content)[:80] if m.content else "")
        )

    async def _invoke():
        return await standard_llm.ainvoke(messages)

    response: AIMessage = await retry_async(
        _invoke,
        max_attempts=retry_config.max_attempts,
        backoff_base=retry_config.backoff_base_seconds,
    )

    return {
        "messages": [response],
        "pending_progress_message": _build_pending_message(response),
    }


async def tool_executor_node(
    state: AgentState,
    tools_by_name: dict,
    slack_notify_func,
    retry_config,
) -> dict:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    # Send progress to Slack before executing tools
    pending = state.get("pending_progress_message")
    if pending and slack_notify_func:
        await slack_notify_func(pending)

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        tool_instance = tools_by_name.get(tool_name)
        if tool_instance is None:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool '{tool_name}' not found.",
                    tool_call_id=tool_call_id,
                )
            )
            continue

        async def _run_tool():
            return await tool_instance.ainvoke(tool_args)

        try:
            result = await retry_async(
                _run_tool,
                max_attempts=retry_config.max_attempts,
                backoff_base=retry_config.backoff_base_seconds,
            )
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call_id)
            )
        except Exception as exc:
            logger.error("Tool '%s' failed after retries: %s", tool_name, exc)
            tool_messages.append(
                ToolMessage(
                    content=f"Tool '{tool_name}' failed: {exc}",
                    tool_call_id=tool_call_id,
                )
            )

    return {
        "messages": tool_messages,
        "pending_progress_message": None,
    }


async def compressor_node(
    state: AgentState,
    light_llm: BaseChatModel,
    cache_store: CacheStore,
    ttl_hours: int,
    retry_config,
) -> dict:
    messages = state["messages"]
    threshold = state.get("compression_threshold", 10000)

    # Find the last ToolMessage(s) to compress
    # We compress messages that exceed the threshold
    updated_messages = []
    new_cache_refs: list[CacheReference] = []

    # Get user query for context
    user_query = ""
    for msg in reversed(messages):
        from langchain_core.messages import HumanMessage
        if isinstance(msg, HumanMessage):
            user_query = str(msg.content)
            break

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            updated_messages.append(msg)
            continue

        content = str(msg.content)
        if len(content.encode()) <= threshold:
            updated_messages.append(msg)
            continue

        # Compress this ToolMessage
        compress_prompt = (
            f"User query: {user_query}\n\nTool result:\n{content}"
        )

        async def _compress():
            return await light_llm.ainvoke([
                SystemMessage(content=_COMPRESSOR_SYSTEM_PROMPT),
                {"role": "user", "content": compress_prompt},
            ])

        try:
            response = await retry_async(
                _compress,
                max_attempts=retry_config.max_attempts,
                backoff_base=retry_config.backoff_base_seconds,
            )
            parsed: CompressorResult = json.loads(str(response.content))
        except Exception as exc:
            logger.error("Compression failed: %s", exc)
            updated_messages.append(msg)
            continue

        focused_summary = parsed.get("focused_summary", content[:500])
        content_index = parsed.get("content_index", "")
        cache_key = parsed.get("cache_key")

        # Save to cache if this came from an MCP tool (has a tool_call_id mapped to a real tool)
        # For simplicity in the skeleton: save whenever cache_key is provided by compressor
        # In full implementation: distinguish MCP vs cache_fetcher calls
        if cache_key and content_index:
            entry = CacheEntry(
                cache_key=cache_key,
                raw_result=content,
                content_index=content_index,
                ttl_hours=ttl_hours,
            )
            cache_store.set(entry)
            new_cache_refs.append(
                CacheReference(
                    cache_key=cache_key,
                    tool_name="",
                    tool_args={},
                    content_index=content_index,
                )
            )

        # 元 ToolMessage の id を引き継ぐことで、add_messages reducer に
        # 「追加」ではなく「置換」と認識させる (checkpointer 利用時の重複防止)
        compressed_msg = ToolMessage(
            content=f"[Compressed]\n{focused_summary}",
            tool_call_id=msg.tool_call_id,
            id=msg.id,
        )
        updated_messages.append(compressed_msg)

    existing_refs = list(state.get("cache_references", []))
    return {
        "messages": updated_messages,
        "cache_references": existing_refs + new_cache_refs,
    }
