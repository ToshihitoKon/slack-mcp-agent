"""_extract_json_object の堅牢な JSON 抽出を検証する。

軽量 LLM がコードフェンスや前置きテキストを付けて返すケースで、
json.loads 直接呼び出しでは "Expecting value: line 1 column 1 (char 0)"
が出ていた問題のリグレッションテスト。
"""

import pytest

from slack_agent.nodes import _extract_json_object


def test_plain_json():
    obj = _extract_json_object('{"focused_summary": "a", "content_index": "b"}')
    assert obj == {"focused_summary": "a", "content_index": "b"}


def test_json_with_surrounding_whitespace():
    obj = _extract_json_object('\n  {"focused_summary": "a", "content_index": "b"}  \n')
    assert obj["focused_summary"] == "a"


def test_json_wrapped_in_code_fence():
    text = '```json\n{"focused_summary": "a", "content_index": "b"}\n```'
    obj = _extract_json_object(text)
    assert obj == {"focused_summary": "a", "content_index": "b"}


def test_json_wrapped_in_bare_code_fence():
    text = '```\n{"focused_summary": "a", "content_index": "b"}\n```'
    obj = _extract_json_object(text)
    assert obj == {"focused_summary": "a", "content_index": "b"}


def test_json_with_leading_explanation():
    text = 'Here is the result:\n{"focused_summary": "a", "content_index": "b"}'
    obj = _extract_json_object(text)
    assert obj == {"focused_summary": "a", "content_index": "b"}


def test_json_with_fence_and_explanation():
    text = (
        "Sure! Here is the summary:\n"
        '```json\n{"focused_summary": "a", "content_index": "b"}\n```\n'
        "Let me know if you need more."
    )
    obj = _extract_json_object(text)
    assert obj["focused_summary"] == "a"
    assert obj["content_index"] == "b"


def test_empty_string_raises():
    # 空応答 (char 0 エラーの典型ケース) は ValueError として扱う
    with pytest.raises(ValueError):
        _extract_json_object("")


def test_non_json_text_raises():
    with pytest.raises(ValueError):
        _extract_json_object("I could not summarize this.")
