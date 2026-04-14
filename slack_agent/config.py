import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

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
    provider: str
    model: str
    credentials: dict


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
class AppConfig:
    slack: SlackConfig
    standard_model: ModelConfig
    light_model: ModelConfig
    retry: RetryConfig
    cache: CacheConfig
    agent: AgentConfig


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
    credentials = models_raw.get("credentials", {})
    standard_model = ModelConfig(provider=provider, model=models_raw["standard"], credentials=credentials)
    light_model = ModelConfig(provider=provider, model=models_raw["light"], credentials=credentials)

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

    return AppConfig(
        slack=slack,
        standard_model=standard_model,
        light_model=light_model,
        retry=retry,
        cache=cache,
        agent=agent,
    )


def build_llm(model_config: ModelConfig) -> BaseChatModel:
    provider = model_config.provider
    model = model_config.model
    creds = model_config.credentials

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=creds["api_key"])

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=creds["api_key"])

    if provider == "bedrock":
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=model,
            region_name=creds.get("region"),
            aws_access_key_id=creds.get("access_key_id"),
            aws_secret_access_key=creds.get("secret_access_key"),
            aws_session_token=creds.get("session_token"),
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=creds.get("base_url", "http://localhost:11434"))

    raise ValueError(f"Unsupported provider: {provider}")
