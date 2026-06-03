# テスト

`slack_agent` パッケージのテスト群。純粋ロジックの単体テストと、LangGraph グラフの統合テストで構成される。

## 実行方法

`pytest` / `pytest-asyncio` は `dev` extras に含まれる。`pyproject.toml` で `asyncio_mode = "auto"` を設定しているため、`async def` のテストにマーカーは不要（既存テストは明示的に `@pytest.mark.asyncio` を付けている）。

```bash
# 全テスト実行
uv run --extra dev pytest -q

# 特定ファイルのみ
uv run --extra dev pytest tests/test_graph_flow.py -q

# 特定テストのみ
uv run --extra dev pytest tests/test_retry.py::test_retry_returns_on_success
```

## テスト一覧

| ファイル | 件数 | 対象 | 主な検証内容 |
|---|---|---|---|
| `test_retry.py` | 9 | `slack_agent/retry.py` | リトライ成功 / transient 失敗からの回復 / max_attempts 到達で last 例外 / 4xx 非リトライ・5xx リトライ判定 / args・kwargs の透過 |
| `test_config.py` | 8 | `slack_agent/config.py` | `${VAR}` / `${VAR:-default}` / 未定義変数の展開 / ネスト構造の再帰展開 / 非文字列値の保持 / `load_config` のフルロードとデフォルト値 |
| `test_thread_history.py` | 8 | `slack_agent/thread_history.py` | API 例外時の空リスト / diff・fallback モードの `oldest` 指定 / subtype・current_ts・空 text の除外 / diff 時のスレッドルート除外と fallback 時の取込 / メンション除去 / assistant メッセージの prefix 付与 |
| `test_graph_routing.py` | 6 | `slack_agent/graph.py` の分岐関数 | `_should_compress` の閾値判定と直近 ToolMessage のみ参照する挙動 / `_after_orchestrator` の tool_calls 有無による分岐 |
| `test_checkpointer.py` | 2 | `slack_agent/checkpointer.py` | `memory` → `MemorySaver` / 未知タイプで `ValueError` |
| `test_cache_store.py` | 10 | `slack_agent/cache.py` | `make_key` の決定性・引数順非依存・ツール名/引数による差分・キー形式 / `is_expired` / `InMemoryCacheStore` の set-get・miss・期限切れエントリの退避 |
| `test_cache_flow.py` | 6 | `slack_agent/nodes.py` (キャッシュフロー) | tool 実行時の決定的 `cache_key` 付与 / `cache_fetcher` 自身の除外 / compressor がメタ情報からキャッシュ保存・参照登録 / 閾値以下は非圧縮・非キャッシュ / 不正 JSON 時の元メッセージ保持 / 非 ToolMessage の素通り |
| `test_graph_flow.py` | 7 | `slack_agent/graph.py` の `build_graph` 全体 | グラフ遷移の統合テスト（下記参照） |
| `test_slack_handler.py` | 6 | `slack_agent/slack_handler.py` の `ThreadLockManager` | 同一 thread の直列化 / 別 thread の並行性 / 使用後の即時解放 / 待機者がいる間の保持 / 例外時の解放 / 未知 key の release 安全性 |

合計 62 テスト。

## グラフ遷移の統合テスト (`test_graph_flow.py`)

LangGraph には専用のテスト API は無く、`build_graph` で compile した実グラフを実行して検証する。

- **ノード遷移順**: `astream(stream_mode="updates")` が各ステップで「実行されたノード名」を更新キーとして返すため、これを収集して遷移順をアサートする。
- **最終 state**: `ainvoke` の戻り値で最終的なメッセージ列・`cache_references`・キャッシュ保存状態を検証する。

検証している経路:

| テスト | 遷移 |
|---|---|
| `test_immediate_answer_ends_without_tools` | `orchestrator → END`（tool_calls 無し） |
| `test_tool_call_small_result_skips_compressor` | `orchestrator → tool_executor → orchestrator`（小結果は compressor を通らない） |
| `test_tool_call_large_result_goes_through_compressor` | `orchestrator → tool_executor → compressor → orchestrator`（閾値超で圧縮経由） |
| `test_large_result_populates_cache_and_references` | 圧縮経由でキャッシュ保存・`cache_references` 登録、ToolMessage が `[Compressed]` に置換 |
| `test_multi_tool_round_trips` | tool を 2 回呼んでから終了する往復 |
| `test_checkpointer_persists_history_across_invocations` | 同一 `thread_id` で履歴が積み上がる（MemorySaver） |
| `test_checkpointer_isolates_distinct_threads` | 異なる `thread_id` の state が混ざらない |

### フェイクの組み方

- `_ScriptedLLM`: `ainvoke` のたびに事前に並べた応答を順に返す。「1 回目は tool_call、2 回目は最終回答」のようにスクリプト化する。`bind_tools` は `self` を返すだけ。
- `_CompressorLLM`: compressor 用 light LLM。常に整形済み JSON を返す。
- `_FakeTool`: 固定の結果文字列を返すツール。結果サイズを変えることで圧縮経路の有無を切り替える。

> `tool_executor` ノードは `langgraph.config.get_config()` を呼ぶが、compiled graph 経由で実行すれば問題なく動く。`slack_notify` は未設定でも `None` になるだけ。

## テスト追加時の指針

- 純粋関数・分岐ロジックは対応するファイル（`test_<module>.py`）に追加する。
- グラフ全体の振る舞いを見る場合は `test_graph_flow.py` にフェイクを差し替えて追加する。
- `retry` を含む経路は `backoff_base=0`（または `RetryConfig(backoff_base_seconds=0)`）で sleep を無効化して高速に保つ。
- `slack_handler.py` は Slack SDK との結合が強くモック量が多いため、イベントハンドラ本体はテスト対象外。ただし `ThreadLockManager` のように SDK 非依存で切り出せるロジックは単体テストする。
