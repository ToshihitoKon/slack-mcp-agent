import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from .retry import retry_async
from .state import AgentState

logger = logging.getLogger(__name__)

_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a helpful assistant with access to various tools.
Answer the user's questions accurately. Use tools when needed.

When you call one or more tools, first write a single short sentence (one line)
in the user's language stating the purpose of what you are about to check or do
(e.g. "ポストモーテムの進捗を確認します"). This sentence becomes the heading shown
above the tool calls, so:
- Write it only ONCE. Do not repeat or restate the same sentence.
- Make it a purposeful sentence about your goal, not a restatement of the tool
  name or its arguments.
"""


def _build_orchestrator_system(extra_prompt: str = "") -> str:
    if extra_prompt.strip():
        return extra_prompt.strip() + "\n\n" + _ORCHESTRATOR_SYSTEM_PROMPT
    return _ORCHESTRATOR_SYSTEM_PROMPT


async def orchestrator_node(
    state: AgentState,
    standard_llm: BaseChatModel,
    retry_config,
    extra_prompt: str = "",
) -> dict:
    system_msg = SystemMessage(content=_build_orchestrator_system(extra_prompt))
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
    }


def _tool_dump(tool_name: str, tool_args: dict) -> str:
    """ツール呼び出しの dump。ツール名と引数を 1 行で表す。

    Plan Block では「目的 (経緯)」を見出し (title) に、その目的のための
    具体的な手段であるツール呼び出しを補足 (details) に置く。
    """
    args_str = json.dumps(tool_args, ensure_ascii=False)
    return f"{tool_name} {args_str}"


async def tool_executor_node(
    state: AgentState,
    tools_by_name: dict,
    progress_reporter,
    retry_config,
) -> dict:
    from .progress import STATUS_COMPLETE, STATUS_IN_PROGRESS

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    # ツールを呼ぶ前のエージェントの思考テキスト (経緯)。これを Plan Block の
    # 見出し (title) に出し、目的のための手段であるツール呼び出しの dump を
    # 補足 (details) に置く。経緯は AIMessage 単位なので先頭タスクの title に使い、
    # 後続タスクはツール名を title にする。
    reasoning = str(last_message.content).strip() if last_message.content else ""

    tool_messages = []
    for idx, tool_call in enumerate(last_message.tool_calls):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        # tool_call 1 件を 1 タスクとして進捗表示する (Issue #6)。
        # title = 経緯 (先頭タスクのみ。無ければツール名)、details = ツール dump。
        if progress_reporter:
            task_title = reasoning if (idx == 0 and reasoning) else tool_name
            await progress_reporter.update_task(
                tool_call_id,
                title=task_title,
                status=STATUS_IN_PROGRESS,
                details=_tool_dump(tool_name, tool_args),
            )

        tool_instance = tools_by_name.get(tool_name)
        if tool_instance is None:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool '{tool_name}' not found.",
                    tool_call_id=tool_call_id,
                )
            )
            if progress_reporter:
                await progress_reporter.update_task(
                    tool_call_id,
                    status=STATUS_COMPLETE,
                    output=f"Tool '{tool_name}' not found.",
                )
            continue

        logger.info("Tool call start: tool=%s args=%r", tool_name, tool_args)

        async def _run_tool():
            return await tool_instance.ainvoke(tool_args)

        try:
            result = await retry_async(
                _run_tool,
                max_attempts=retry_config.max_attempts,
                backoff_base=retry_config.backoff_base_seconds,
            )
            logger.info("Tool call done: tool=%s result_size=%dB", tool_name, len(str(result).encode()))
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                )
            )
            if progress_reporter:
                await progress_reporter.update_task(
                    tool_call_id, status=STATUS_COMPLETE
                )
        except Exception as exc:
            logger.error("Tool '%s' failed after retries: %s", tool_name, exc)
            tool_messages.append(
                ToolMessage(
                    content=f"Tool '{tool_name}' failed: {exc}",
                    tool_call_id=tool_call_id,
                )
            )
            if progress_reporter:
                await progress_reporter.update_task(
                    tool_call_id,
                    status=STATUS_COMPLETE,
                    output=f"failed: {exc}",
                )

    return {
        "messages": tool_messages,
    }
