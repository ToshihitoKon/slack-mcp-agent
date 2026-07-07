# slack-mcp-agent

LangGraph で実装した Slack 向けエージェント。ユーザーの入力に応じて MCP ツールを呼び出し、ツール結果を必要に応じて圧縮・キャッシュしながら回答を生成する。Slack の DM とメンションに反応する。

## 特徴

- **MCP ツール連携** — `mcp_config.json` に定義した stdio MCP サーバーのツールを LangGraph のツールとして自動ロードする。
- **ツール結果の圧縮とキャッシュ** — 大きなツール結果は軽量モデルで要約し、生の結果はキャッシュに退避する。後続のやり取りでは `cache_fetcher` ツールで生の結果を取り出せる。
- **スレッド単位の会話履歴** — Slack の `thread_ts` を LangGraph の `thread_id` として扱い、checkpointer で会話履歴を永続化する。
- **進捗表示** — ツール実行を Slack の Plan Block（`chat.startStream` ベースの構造化タスク表示）で逐次通知する。Plan Block 非対応の環境では自動でテキスト表示にフォールバックする。
- **リトライ** — ツール呼び出し・LLM 呼び出しを指数バックオフでリトライする（5xx はリトライ、4xx はリトライしない）。

## アーキテクチャ

LangGraph の 3 ノードで構成される。

| ノード | 役割 |
|---|---|
| `orchestrator` | 標準モデル。思考・ツール呼び出し判断・最終回答生成 |
| `tool_executor` | MCP ツールまたはビルトインツール（`cache_fetcher`）を実行 |
| `compressor` | ToolMessage のサイズ判定・軽量モデルで圧縮・キャッシュ保存 |

### グラフ遷移

```
orchestrator ──→ END                          # tool_calls なし
     │
     └──→ tool_executor ──→ compressor ──→ orchestrator   # 直近 ToolMessage が閾値超え
                    │
                    └──────────────────→ orchestrator     # 閾値未満
```

### キャッシュの流れ

1. `tool_executor` がツール実行時に `cache_key`（`tool_name` + 引数のハッシュ）を決定的に生成し、ToolMessage のメタ情報に格納する。
2. `compressor` が閾値超えの ToolMessage を要約に置換しつつ、生の結果を `cache_key` でキャッシュへ保存し、`cache_references` に登録する。
3. `orchestrator` は `cache_references` をシステムプロンプトに差し込み、必要なら `cache_fetcher` ツールで生の結果を取り出す。

## セットアップ

### 必要環境

- Python 3.14 以上
- [uv](https://github.com/astral-sh/uv)（依存解決と MCP サーバー起動に使う `uvx` を含む）

### 依存インストール

```bash
uv sync
```

LLM プロバイダに応じて extras を追加する。

```bash
uv sync --extra openai      # OpenAI
uv sync --extra anthropic   # Anthropic
uv sync --extra bedrock     # AWS Bedrock
uv sync --extra ollama      # Ollama
```

## 設定

### `settings.json`

アプリ全体の設定。`${VAR}` / `${VAR:-default}` 形式で環境変数を展開できる。

```json
{
  "slack": {
    "bot_token": "${SLACK_BOT_TOKEN}",
    "app_token": "${SLACK_APP_TOKEN}",
    "allowed_user_ids": []
  },
  "models": {
    "provider": "openai",
    "standard": "gpt-5-nano-2025-08-07",
    "light": "gpt-5-nano-2025-08-07",
    "options": {}
  },
  "retry": { "max_attempts": 3, "backoff_base_seconds": 1.0 },
  "cache": { "ttl_hours": 6 },
  "agent": { "compression_threshold_bytes": 10000, "recursion_limit": 25, "progress_mode": "auto" },
  "storage": { "type": "memory" }
}
```

| キー | 説明 |
|---|---|
| `slack.bot_token` / `slack.app_token` | Slack Bot Token / App-Level Token（Socket Mode 用） |
| `slack.allowed_user_ids` | 応答を許可するユーザー ID。空配列なら全員許可 |
| `models.provider` | LLM プロバイダ（`openai` / `anthropic` / `bedrock` / `ollama` など） |
| `models.standard` / `models.light` | orchestrator 用 / compressor 用のモデル名 |
| `models.options` | `init_chat_model` にそのまま渡す追加パラメータ（プロバイダ固有） |
| `retry` | リトライ回数とバックオフ基準秒数 |
| `cache.ttl_hours` | キャッシュエントリの有効期限（時間） |
| `agent.compression_threshold_bytes` | ToolMessage を圧縮する閾値（バイト） |
| `agent.recursion_limit` | LangGraph の再帰上限 |
| `agent.progress_mode` | 進捗表示モード。`auto`（Plan Block を試し失敗で text にフォールバック）/ `plan` / `text` |
| `storage.type` | checkpointer のバックエンド。現状は `memory` のみ対応 |

#### Anthropic のモデル（Bedrock 経由含む）を使う場合

このアプリは応答の content block を `str(message.content)` で単純にテキスト化している
（[`slack_agent/nodes.py`](slack_agent/nodes.py) / [`slack_agent/slack_handler.py`](slack_agent/slack_handler.py)）。
拡張思考 (thinking) を返すモデルは content が `[{"type": "thinking", ...}, {"type": "text", "text": "..."}]`
のような list になり、この単純な `str()` では thinking の signature を含む Python の repr が
そのまま出力されてしまう。Claude Sonnet 5 のような拡張思考対応モデルは `options` を空にしていても
デフォルトで thinking を返すため、明示的に無効化する必要がある。

`models.provider` に `bedrock` を指定すると `langchain.chat_models.init_chat_model` は
Bedrock **Invoke API** を使うレガシークラス（`langchain_aws.chat_models.ChatBedrock`）を解決する
（Bedrock **Converse API** 用の `ChatBedrockConverse` ではない）。そのため Converse API 向けの
`additional_model_request_fields` はここでは効かず、`ValidationException: additional_model_request_fields:
Extra inputs are not permitted` になる。`model_kwargs` 経由でリクエストボディに直接 `thinking` を
渡すこと。

```json
"models": {
  "provider": "bedrock",
  "standard": "global.anthropic.claude-sonnet-5",
  "options": {
    "model_kwargs": {
      "thinking": {"type": "disabled"}
    }
  }
}
```

### `mcp_config.json`

接続する MCP サーバーを定義する。`mcpServers` の各エントリは `command` / `args` / `env` を持つ。

```json
{
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  }
}
```

### `prompt.md`

存在する場合、その内容が orchestrator のシステムプロンプト先頭に追記される（任意）。

## 実行

```bash
# 環境変数をセット
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_APP_TOKEN=xapp-...
export OPENAI_API_KEY=sk-...   # 利用プロバイダに応じて

# 起動
uv run python main.py

# MCP サーバー接続だけ検証して終了（ドライラン）
uv run python main.py --mcp-check

# 設定ファイルのパスを変える
uv run python main.py --settings settings.json --mcp-config mcp_config.json
```

`DEBUG_LLM=1` を付けると LangChain の verbose/debug が有効になり、LLM への入出力がすべて出力される。

```bash
DEBUG_LLM=1 uv run python main.py
```

## Docker

```bash
docker build -t slack-mcp-agent .
docker run --rm \
  -e SLACK_BOT_TOKEN -e SLACK_APP_TOKEN -e OPENAI_API_KEY \
  -v "$PWD/settings.json:/app/settings.json" \
  -v "$PWD/mcp_config.json:/app/mcp_config.json" \
  slack-mcp-agent
```

## テスト

```bash
uv run --extra dev pytest -q
```

テストスイートの詳細は [`tests/README.md`](tests/README.md) を参照。

## ディレクトリ構成

```
.
├── main.py          # エントリポイント（CLI 引数・ドライラン・起動）
├── settings.json    # アプリ設定
├── mcp_config.json  # MCP サーバー定義
├── prompt.md        # orchestrator への追加システムプロンプト（任意）
├── Dockerfile
├── slack_agent/     # エージェント本体（ノード・グラフ・ツール・設定など）
└── tests/           # テスト（tests/README.md 参照）
```
