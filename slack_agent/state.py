from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class CacheReference(TypedDict):
    cache_key: str
    tool_name: str
    tool_args: dict
    content_index: str


class CompressorResult(TypedDict):
    focused_summary: str
    content_index: str


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    compression_threshold: int
    cache_references: list[CacheReference]
    # 最後に bot が postMessage した ts。次回起動時の差分取得 (oldest) に使う
    last_bot_response_ts: str | None
