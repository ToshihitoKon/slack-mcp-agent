import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def _expand_env_vars(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} patterns in a string."""
    def replacer(match: re.Match) -> str:
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
            return os.environ.get(var_name, default)
        return os.environ.get(var_expr, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replacer, value)


def _expand_recursive(obj):
    """Recursively expand env vars in all string values of a dict/list."""
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _expand_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_recursive(v) for v in obj]
    return obj


@dataclass
class SlackConfig:
    bot_token: str
    app_token: str
    allowed_user_ids: list[str]


@dataclass
class ModelConfig:
    model: str
    options: dict


@dataclass
class RetryConfig:
    max_attempts: int
    backoff_base_seconds: float


@dataclass
class CacheConfig:
    ttl_hours: int


@dataclass
class AgentConfig:
    compression_threshold_bytes: int
    recursion_limit: int


@dataclass
class StorageConfig:
    """LangGraph checkpointer のバックエンド設定。

    type: "memory" のみ対応。将来 sqlite/postgres を追加予定。
    """
    type: str


@dataclass
class AppConfig:
    slack: SlackConfig
    standard_model: ModelConfig
    light_model: ModelConfig
    retry: RetryConfig
    cache: CacheConfig
    agent: AgentConfig
    storage: StorageConfig


def load_config(settings_path: str = "settings.json") -> AppConfig:
    raw = json.loads(Path(settings_path).read_text())
    raw = _expand_recursive(raw)

    slack_raw = raw["slack"]
    slack = SlackConfig(
        bot_token=slack_raw["bot_token"],
        app_token=slack_raw["app_token"],
        allowed_user_ids=slack_raw.get("allowed_user_ids", []),
    )

    models_raw = raw["models"]
    provider = models_raw["provider"]
    model_options = models_raw.get("options", {})
    standard_model = ModelConfig(model=f"{provider}:{models_raw['standard']}", options=model_options)
    light_model = ModelConfig(model=f"{provider}:{models_raw['light']}", options=model_options)

    retry_raw = raw.get("retry", {})
    retry = RetryConfig(
        max_attempts=retry_raw.get("max_attempts", 3),
        backoff_base_seconds=retry_raw.get("backoff_base_seconds", 1.0),
    )

    cache_raw = raw.get("cache", {})
    cache = CacheConfig(ttl_hours=cache_raw.get("ttl_hours", 6))

    agent_raw = raw.get("agent", {})
    agent = AgentConfig(
        compression_threshold_bytes=agent_raw.get("compression_threshold_bytes", 10000),
        recursion_limit=agent_raw.get("recursion_limit", 25),
    )

    storage_raw = raw.get("storage", {"type": "memory"})
    storage = StorageConfig(type=storage_raw.get("type", "memory"))

    return AppConfig(
        slack=slack,
        standard_model=standard_model,
        light_model=light_model,
        retry=retry,
        cache=cache,
        agent=agent,
        storage=storage,
    )


def build_llm(model_config: ModelConfig) -> BaseChatModel:
    return init_chat_model(model_config.model, **model_config.options)
