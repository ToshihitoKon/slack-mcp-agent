"""環境変数展開と settings.json ロードの挙動を検証する。"""

import json

from slack_agent.config import (
    AppConfig,
    _expand_env_vars,
    _expand_recursive,
    load_config,
)


def test_expand_env_vars_substitutes_defined(monkeypatch):
    monkeypatch.setenv("MY_VAR", "value123")
    assert _expand_env_vars("${MY_VAR}") == "value123"
    assert _expand_env_vars("prefix-${MY_VAR}-suffix") == "prefix-value123-suffix"


def test_expand_env_vars_uses_default_when_undefined(monkeypatch):
    monkeypatch.delenv("MISSING", raising=False)
    assert _expand_env_vars("${MISSING:-fallback}") == "fallback"


def test_expand_env_vars_prefers_env_over_default(monkeypatch):
    monkeypatch.setenv("PRESENT", "real")
    assert _expand_env_vars("${PRESENT:-fallback}") == "real"


def test_expand_env_vars_leaves_undefined_without_default(monkeypatch):
    monkeypatch.delenv("UNSET", raising=False)
    # default 無し・未定義の場合は元の ${...} 文字列のまま残す
    assert _expand_env_vars("${UNSET}") == "${UNSET}"


def test_expand_recursive_handles_nested_structures(monkeypatch):
    monkeypatch.setenv("TOKEN", "secret")
    obj = {
        "a": "${TOKEN}",
        "b": ["${TOKEN}", "plain"],
        "c": {"d": "${TOKEN}", "e": 42},
    }
    result = _expand_recursive(obj)
    assert result == {
        "a": "secret",
        "b": ["secret", "plain"],
        "c": {"d": "secret", "e": 42},
    }


def test_expand_recursive_preserves_non_string_values():
    obj = {"num": 1, "flag": True, "none": None, "list": [1, 2]}
    assert _expand_recursive(obj) == obj


def _write_settings(tmp_path, data: dict) -> str:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_load_config_full(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "xoxb-test")
    settings = {
        "slack": {
            "bot_token": "${BOT_TOKEN}",
            "app_token": "xapp-test",
            "allowed_user_ids": ["U1", "U2"],
        },
        "models": {
            "provider": "anthropic",
            "standard": "claude-opus",
            "options": {"temperature": 0},
        },
        "retry": {"max_attempts": 5, "backoff_base_seconds": 2.0},
        "agent": {
            "recursion_limit": 30,
            "progress_mode": "plan",
            "mcp_tool_timeout_seconds": 120,
        },
        "storage": {"type": "memory"},
    }
    cfg = load_config(_write_settings(tmp_path, settings))

    assert isinstance(cfg, AppConfig)
    assert cfg.slack.bot_token == "xoxb-test"
    assert cfg.slack.allowed_user_ids == ["U1", "U2"]
    assert cfg.standard_model.model == "anthropic:claude-opus"
    assert cfg.standard_model.options == {"temperature": 0}
    assert cfg.retry.max_attempts == 5
    assert cfg.retry.backoff_base_seconds == 2.0
    assert cfg.agent.recursion_limit == 30
    assert cfg.agent.progress_mode == "plan"
    assert cfg.agent.mcp_tool_timeout_seconds == 120
    assert cfg.storage.type == "memory"


def test_load_config_applies_defaults(tmp_path):
    settings = {
        "slack": {"bot_token": "b", "app_token": "a"},
        "models": {"provider": "openai", "standard": "gpt"},
    }
    cfg = load_config(_write_settings(tmp_path, settings))

    # 省略時のデフォルト値
    assert cfg.slack.allowed_user_ids == []
    assert cfg.retry.max_attempts == 3
    assert cfg.retry.backoff_base_seconds == 1.0
    assert cfg.agent.recursion_limit == 25
    assert cfg.agent.progress_mode == "auto"
    assert cfg.agent.mcp_tool_timeout_seconds == 60.0
    assert cfg.storage.type == "memory"
