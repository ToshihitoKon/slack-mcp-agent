from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # 最後に bot が postMessage した ts。次回起動時の差分取得 (oldest) に使う
    last_bot_response_ts: str | None
