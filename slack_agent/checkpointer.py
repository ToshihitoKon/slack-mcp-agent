"""LangGraph checkpointer のファクトリ。

Slack の thread_ts を thread_id として AgentState 全体を保存・復元するため、
LangGraph の checkpointer を生成するファクトリを提供する。

将来 sqlite/postgres バックエンドを追加する際は:
- create_checkpointer の dispatch に case を追加する
- 依存パッケージは optional-dependencies として追加する
  (例: langgraph-checkpoint-sqlite, langgraph-checkpoint-postgres)
- StorageConfig に必要なフィールド (path, dsn 等) を追加する
"""

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from .config import StorageConfig

logger = logging.getLogger(__name__)


def create_checkpointer(cfg: StorageConfig) -> BaseCheckpointSaver:
    """StorageConfig に基づいて checkpointer を生成する。

    現状は "memory" のみ対応。未知のタイプは ValueError を投げる。
    """
    if cfg.type == "memory":
        logger.info("Using MemorySaver checkpointer (in-memory, non-persistent)")
        return MemorySaver()
    # 将来の拡張ポイント:
    # if cfg.type == "sqlite":
    #     from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    #     return AsyncSqliteSaver.from_conn_string(cfg.path)
    # if cfg.type == "postgres":
    #     from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    #     return AsyncPostgresSaver.from_conn_string(cfg.dsn)
    raise ValueError(f"Unknown storage type: {cfg.type!r}")
