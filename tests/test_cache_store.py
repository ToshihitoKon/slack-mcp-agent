"""CacheStore.make_key / CacheEntry.is_expired / InMemoryCacheStore を検証する。"""

from datetime import datetime, timedelta

from slack_agent.cache import CacheEntry, CacheStore, InMemoryCacheStore


def test_make_key_is_deterministic():
    k1 = CacheStore.make_key("tool", {"a": 1, "b": 2})
    k2 = CacheStore.make_key("tool", {"a": 1, "b": 2})
    assert k1 == k2


def test_make_key_is_independent_of_arg_order():
    # sort_keys=True により引数の順序に依存しない
    k1 = CacheStore.make_key("tool", {"a": 1, "b": 2})
    k2 = CacheStore.make_key("tool", {"b": 2, "a": 1})
    assert k1 == k2


def test_make_key_differs_by_tool_name():
    assert CacheStore.make_key("toolA", {"x": 1}) != CacheStore.make_key("toolB", {"x": 1})


def test_make_key_differs_by_args():
    assert CacheStore.make_key("tool", {"x": 1}) != CacheStore.make_key("tool", {"x": 2})


def test_make_key_format():
    key = CacheStore.make_key("my_tool", {"q": "hi"})
    name, _, digest = key.partition(":")
    assert name == "my_tool"
    assert len(digest) == 16


def test_entry_not_expired_when_ttl_positive():
    entry = CacheEntry("k", "raw", "idx", ttl_hours=1)
    assert entry.is_expired() is False


def test_entry_expired_when_past_expiry():
    entry = CacheEntry("k", "raw", "idx", ttl_hours=1)
    # 期限を過去に書き換える
    entry.expires_at = datetime.utcnow() - timedelta(seconds=1)
    assert entry.is_expired() is True


def test_store_set_and_get_roundtrip():
    store = InMemoryCacheStore()
    entry = CacheEntry("k1", "raw-data", "idx", ttl_hours=1)
    store.set(entry)
    got = store.get("k1")
    assert got is entry
    assert got.raw_result == "raw-data"


def test_store_get_returns_none_on_miss():
    store = InMemoryCacheStore()
    assert store.get("missing") is None


def test_store_get_evicts_expired_entry():
    store = InMemoryCacheStore()
    entry = CacheEntry("k1", "raw", "idx", ttl_hours=1)
    entry.expires_at = datetime.utcnow() - timedelta(seconds=1)
    store.set(entry)

    assert store.get("k1") is None
    # 期限切れエントリは内部ストアからも削除される
    assert "k1" not in store._store
