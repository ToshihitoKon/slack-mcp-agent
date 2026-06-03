import json
import logging
import re
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

# ```json ... ``` や ``` ... ``` で囲まれたコードフェンスを剥がす
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json_object(text: str) -> dict:
    """LLM 応答テキストから JSON オブジェクトを堅牢に抽出する。

    軽量モデルはコードフェンスや前置きテキストを付けることがあり、
    json.loads をそのまま掛けると失敗する。以下の順で復旧を試みる:
      1. そのまま json.loads
      2. コードフェンスを剥がして json.loads
      3. 最初の { から最後の } までを抜き出して json.loads
    いずれも失敗した場合は ValueError を送出する。
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    stripped = _CODE_FENCE_RE.sub("", text).strip()
    if stripped != text:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"No JSON object found in response: {text[:200]!r}")


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

        # tool 実行時に決定的な cache_key を生成し、ToolMessage に
        # メタ情報として持たせる。compressor_node がこれを使ってキャッシュ保存する。
        # cache_fetcher 自身の結果はキャッシュ対象外 (cache_key を付けない)。
        cache_meta: dict[str, Any] = {}
        if tool_name != "cache_fetcher":
            cache_meta = {
                "cache_key": CacheStore.make_key(tool_name, tool_args),
                "cache_tool_name": tool_name,
                "cache_tool_args": tool_args,
            }

        try:
            result = await retry_async(
                _run_tool,
                max_attempts=retry_config.max_attempts,
                backoff_base=retry_config.backoff_base_seconds,
            )
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                    additional_kwargs=cache_meta,
                )
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

    # 構造化出力でモデルに JSON を強制する。未対応モデルの場合は
    # 生テキストを _extract_json_object でフォールバックパースする。
    try:
        structured_llm = light_llm.with_structured_output(CompressorResult)
    except (NotImplementedError, AttributeError):
        structured_llm = None

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

        compress_messages = [
            SystemMessage(content=_COMPRESSOR_SYSTEM_PROMPT),
            {"role": "user", "content": compress_prompt},
        ]

        async def _compress_structured():
            # with_structured_output は dict (CompressorResult) を直接返す
            return await structured_llm.ainvoke(compress_messages)

        async def _compress_text():
            response = await light_llm.ainvoke(compress_messages)
            return _extract_json_object(str(response.content))

        try:
            if structured_llm is not None:
                parsed: CompressorResult = await retry_async(
                    _compress_structured,
                    max_attempts=retry_config.max_attempts,
                    backoff_base=retry_config.backoff_base_seconds,
                )
            else:
                parsed = await retry_async(
                    _compress_text,
                    max_attempts=retry_config.max_attempts,
                    backoff_base=retry_config.backoff_base_seconds,
                )
        except Exception as exc:
            # 構造化出力が実行時に失敗した場合も生テキスト抽出にフォールバック
            if structured_llm is not None:
                logger.warning(
                    "Structured compression failed (%s); falling back to text parsing",
                    exc,
                )
                try:
                    parsed = await retry_async(
                        _compress_text,
                        max_attempts=retry_config.max_attempts,
                        backoff_base=retry_config.backoff_base_seconds,
                    )
                except Exception as exc2:
                    logger.error("Compression failed: %s", exc2)
                    updated_messages.append(msg)
                    continue
            else:
                logger.error("Compression failed: %s", exc)
                updated_messages.append(msg)
                continue

        focused_summary = parsed.get("focused_summary", content[:500])
        content_index = parsed.get("content_index", "")

        # cache_key は tool_executor_node が決定的に生成し ToolMessage の
        # additional_kwargs に格納している。LLM 任せにしない。
        meta = msg.additional_kwargs or {}
        cache_key = meta.get("cache_key")

        # MCP tool 由来 (cache_key 付き) の結果だけをキャッシュ保存する。
        if cache_key:
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
                    tool_name=meta.get("cache_tool_name", ""),
                    tool_args=meta.get("cache_tool_args", {}),
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
