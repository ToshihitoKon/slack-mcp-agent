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
    cache_key: str | None


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    compression_threshold: int
    cache_references: list[CacheReference]
    pending_progress_message: str | None
