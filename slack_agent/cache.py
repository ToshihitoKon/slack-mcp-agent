import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any


class CacheEntry:
    def __init__(self, cache_key: str, raw_result: str, content_index: str, ttl_hours: int):
        self.cache_key = cache_key
        self.raw_result = raw_result
        self.content_index = content_index
        self.created_at = datetime.utcnow()
        self.expires_at = self.created_at + timedelta(hours=ttl_hours)

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


class CacheStore(ABC):
    @abstractmethod
    def get(self, cache_key: str) -> CacheEntry | None:
        ...

    @abstractmethod
    def set(self, entry: CacheEntry) -> None:
        ...

    @staticmethod
    def make_key(tool_name: str, tool_args: dict) -> str:
        args_hash = hashlib.sha256(
            json.dumps(tool_args, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"{tool_name}:{args_hash}"


class InMemoryCacheStore(CacheStore):
    def __init__(self):
        self._store: dict[str, CacheEntry] = {}

    def get(self, cache_key: str) -> CacheEntry | None:
        entry = self._store.get(cache_key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._store[cache_key]
            return None
        return entry

    def set(self, entry: CacheEntry) -> None:
        self._store[entry.cache_key] = entry
