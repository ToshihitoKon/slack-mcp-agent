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
When making tool calls, also include a brief progress message in the format:
<progress>Brief description of what you are doing</progress>
"""

_COMPRESSOR_SYSTEM_PROMPT = """\
You are a summarization assistant.
Given a tool result and the user's query, produce:
1. focused_summary: a concise summary relevant to the user's query
2. content_index: a structural index of the full result for future reference

Respond ONLY with valid JSON matching:
{"focused_summary": "...", "content_index": "..."}
"""


def _build_orchestrator_system(cache_references: list[CacheReference]) -> str:
    prompt = _ORCHESTRATOR_SYSTEM_PROMPT
    if cache_references:
        refs_text = "\n".join(
            f"- cache_key={r['cache_key']} tool={r['tool_name']} index={r['content_index']}"
            for r in cache_references
        )
        prompt += f"\n\nAvailable cached results (use cache_fetcher tool to retrieve):\n{refs_text}"
    return prompt


def _extract_progress_message(ai_message: AIMessage) -> str | None:
    content = ai_message.content
    if not isinstance(content, str):
        return None
    import re
    m = re.search(r"<progress>(.*?)</progress>", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


async def orchestrator_node(
    state: AgentState,
    standard_llm: BaseChatModel,
    retry_config,
) -> dict:
    system_msg = SystemMessage(content=_build_orchestrator_system(state.get("cache_references", [])))
    messages = [system_msg] + list(state["messages"])

    async def _invoke():
        return await standard_llm.ainvoke(messages)

    response: AIMessage = await retry_async(
        _invoke,
        max_attempts=retry_config.max_attempts,
        backoff_base=retry_config.backoff_base_seconds,
    )

    pending = None
    if response.tool_calls:
        pending = _extract_progress_message(response)
        if pending is None:
            pending = "Processing..."

    return {
        "messages": [response],
        "pending_progress_message": pending,
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

        compressed_msg = ToolMessage(
            content=f"[Compressed]\n{focused_summary}",
            tool_call_id=msg.tool_call_id,
        )
        updated_messages.append(compressed_msg)

    existing_refs = list(state.get("cache_references", []))
    return {
        "messages": updated_messages,
        "cache_references": existing_refs + new_cache_refs,
    }
