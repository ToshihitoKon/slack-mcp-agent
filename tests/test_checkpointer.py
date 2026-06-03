"""create_checkpointer のバックエンド選択を検証する。"""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from slack_agent.checkpointer import create_checkpointer
from slack_agent.config import StorageConfig


def test_memory_backend_returns_memory_saver():
    cp = create_checkpointer(StorageConfig(type="memory"))
    assert isinstance(cp, MemorySaver)


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unknown storage type"):
        create_checkpointer(StorageConfig(type="redis"))
