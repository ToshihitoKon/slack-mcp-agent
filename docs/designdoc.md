# Slack Agent Design Doc

## 概要

LangGraph（Python）を用いたSlack向けエージェント。ユーザーの入力に応じてMCPツールを呼び出し、結果を必要に応じて圧縮・キャッシュしながら回答を生成する。

---

## グラフ構成

### ノード

| ノード | 役割 |
|---|---|
| `orchestrator` | 標準モデル。思考・ToolCall判断・最終回答生成 |
| `tool_executor` | MCPツールまたはビルトインツール（`cache_fetcher`）を実行 |
| `compressor` | ToolMessageのサイズ判定・軽量モデルで圧縮・キャッシュ保存 |

### エッジ条件

```
orchestrator  → END              # tool_callsなし
orchestrator  → tool_executor    # tool_callsあり

tool_executor → compressor       # 直近ToolMessageがしきい値超え
tool_executor → orchestrator     # しきい値未満

compressor    → orchestrator     # 常に
```

### グラフ図

```
orchestrator ──→ END
     │
     └──→ tool_executor ──→ compressor ──→ orchestrator
                    │
                    └──────────────────→ orchestrator
```

---

## State

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    compression_threshold: int
    cache_references: list[CacheReference]
    pending_progress_message: str | None
```

### CacheReference

```python
class CacheReference(TypedDict):
    cache_key: str       # tool_name + hash(args)
    tool_name: str
    tool_args: dict
    content_index: str   # orchestratorがキャッシュの中身を把握するためのindex
```

---

## ノード詳細

### orchestrator

**input（依存）**
- `messages` — 会話履歴全体
- `cache_references` — 利用可能なキャッシュ一覧（system promptに動的に差し込む）

**output**
- `messages` に追加: AIMessage（tool_calls or 最終回答）
- `pending_progress_message` — tool_callsがある場合のみ生成

**備考**
- モデル: 標準モデル（`claude-sonnet-4-5`）
- `cache_references`はsystem promptの末尾に差し込む
- tool_callsと同時に進捗メッセージを生成することでLLM呼び出しを増やさない
- 全ツールが失敗した場合でも、失敗した旨のコンテキストをもとにレスポンスを生成する

---

### tool_executor

**input（依存）**
- `messages` — 直近のAIMessageからtool_callsを取得
- `pending_progress_message` — Slack送信に使用（実行前に送信して破棄）

**output**
- `messages` に追加: ToolMessage（raw result または失敗情報）
- `pending_progress_message` をNoneにクリア

**ツールの種類**
- MCPツール: 外部MCP serverを呼び出す
- ビルトインツール（`cache_fetcher`）: ローカルキャッシュからraw resultを取得

**リトライ**
- 失敗時はexponential backoffでリトライ
- リトライ回数は `settings.json` で指定（デフォルト: 3回）
- リトライ対象: MCPツール呼び出しおよびLLM API呼び出し
- HTTP Status codeを加味する（5xx はリトライ、4xx はリトライしない）
- 全リトライ失敗時はToolMessageに失敗情報を含めてorchestratorに渡す

**備考**
- `pending_progress_message`はtool実行前にSlack通知モジュールへ渡す
- Slack通知はLangGraphノードではなく単純な関数として実装

---

### compressor

**input（依存）**
- `messages` — 直近のToolMessageとuser queryを取得
- `compression_threshold` — サイズ判定用（バイト数で指定）

**output**
- `messages` の直近ToolMessageを上書き
  - `focused_summary`: 現在のuser queryの観点に沿った要約
  - `content_index`: raw result全体の構造・内容のindex（再参照判断用）
- `cache_references` に追加（`cache_key != null`の場合のみ）

**CompressorResult**
```python
class CompressorResult(TypedDict):
    focused_summary: str
    content_index: str
    cache_key: str | None
```

**cache_keyの有無による振る舞い**

| 呼び出し元 | キャッシュ保存 | index生成 | cache_key |
|---|---|---|---|
| MCPツール経由 | ✅ | ✅ | 生成して返す |
| cache_fetcher経由 | ❌ | ❌ | null |

**備考**
- モデル: 軽量モデル（`claude-haiku-4-5`）
- `cache_key = null`の場合、`cache_references`への追加とsystem promptへのindex差し込みをスキップ

---

## キャッシュ

### ストレージ

- インメモリ（dict）で実装
- キャッシュストアはクラスまたはwrapperで抽象化し、将来的にRedisへ差し替えられるようにする

### キャッシュエントリ

```python
{
    "cache_key": "tool_name:hash(args)",
    "raw_result": "...",
    "content_index": "...",
    "expires_at": now + timedelta(hours=N),
    "created_at": now
}
```

### キャッシュの有効期限管理

- 有効期限は `settings.json` で指定（デフォルト: 6時間）
- 有効期限の判定はキャッシュストア側の責務
- orchestratorはexpires_atを保持しない
- cache_fetcherはキャッシュミス・期限切れの場合はToolMessageにその旨を返す

---

## Slack通知モジュール

LangGraphのノードではなく独立したモジュールとして実装。

### トリガー

- DM（ダイレクトメッセージ）
- メンション（`@bot`）

### 進捗メッセージ

- スレッドの返信1件を作成し、以降は同メッセージを**更新**していく
  - 複数投稿はノイズになるため、1メッセージに集約する
- `pending_progress_message` の内容でSlack APIのupdate APIを呼ぶ

### 最終回答

- 一括送信（ストリーミングなし）
- スレッドに返信する形で送信

### 呼び出しタイミング

- tool_executor実行前: `pending_progress_message`を送信（進捗メッセージを更新）
- 最終回答生成後: 回答本文をスレッドに送信

### 責務

- LangGraphのStateには関与しない
- Slack Bolt（async）経由で送信
- Socket Modeで接続

---

## MCPツール設定

### 設定ファイル

- `mcp_config.json` に接続するMCPサーバーを記載
- フォーマットはClaude Codeと互換性のある基本仕様に従う
- 複数サーバーを同時に使用可能

### フォーマット例（Claude Code互換）

```json
{
  "mcpServers": {
    "server_name": {
      "command": "uvx",
      "args": ["mcp-server-name"],
      "env": {
        "API_KEY": "..."
      }
    }
  }
}
```

### 通信プロトコル

- stdio（サブプロセス起動）およびHTTP/SSEをサポート
- デファクトスタンダードなライブラリ（例: `mcp` パッケージ）を使い、極力簡潔に実装する

---

## モデル設定

### LLMプロバイダー抽象化

LangChainの `BaseChatModel` を通じて複数のLLMプロバイダーを統一インターフェースで扱う。対応プロバイダー：

| プロバイダー | LangChainクラス | パッケージ |
|---|---|---|
| Anthropic | `ChatAnthropic` | `langchain-anthropic` |
| OpenAI互換 | `ChatOpenAI` | `langchain-openai` |
| AWS Bedrock | `ChatBedrock` | `langchain-aws` |
| Ollama | `ChatOllama` | `langchain-ollama` |

ノード内では `BaseChatModel` として受け取り、プロバイダー固有のコードを持たない。

### settings.json によるモデル指定

```json
{
  "models": {
    "provider": "anthropic",
    "standard": "claude-sonnet-4-5",
    "light": "claude-haiku-4-5",
    "credentials": {
      "api_key": "${ANTHROPIC_API_KEY}"
    }
  }
}
```

- `models.provider`: 使用するLLMプロバイダー（全モデル共通）
- `models.standard`: orchestratorが使用するモデル名
- `models.light`: compressorが使用するモデル名
- `models.credentials`: プロバイダーへの認証情報。`config.py` が解釈してLangChainクラスに明示的に渡す
- `provider` に応じて `config.py` が適切なLangChainクラスをインスタンス化する

#### credentialsのプロバイダー別フィールド

| プロバイダー | フィールド |
|---|---|
| `anthropic` | `api_key` |
| `openai` | `api_key` |
| `bedrock` | `region`、`access_key_id`、`secret_access_key`、`session_token`（省略可） |
| `ollama` | `base_url`（省略時: `http://localhost:11434`） |

---

## settings.json

アプリケーション全体の設定を一元管理するファイル。

```json
{
  "anthropic_api_key": "${ANTHROPIC_API_KEY}",
  "slack": {
    "bot_token": "${SLACK_BOT_TOKEN}",
    "app_token": "${SLACK_APP_TOKEN}",
    "allowed_user_ids": []
  },
  "retry": {
    "max_attempts": 3,
    "backoff_base_seconds": 1.0
  },
  "cache": {
    "ttl_hours": 6
  },
  "agent": {
    "compression_threshold_bytes": 10000,
    "recursion_limit": 25
  }
}
```

- `anthropic_api_key`: Anthropic APIキー
- `slack.bot_token`: Slack Bot Token（`xoxb-...`）
- `slack.app_token`: Slack App Token（Socket Mode用、`xapp-...`）
- `retry.max_attempts`: リトライ最大回数（MCPツール・LLM API共通）
- `retry.backoff_base_seconds`: exponential backoffの基底秒数
- `cache.ttl_hours`: キャッシュの有効期限（時間）
- `agent.compression_threshold_bytes`: ToolMessage圧縮のしきい値（バイト数）
- `agent.recursion_limit`: LangGraphの再帰上限
- `slack.allowed_user_ids`: 利用を許可するSlackユーザーID（空リストの場合は全員許可）

### 環境変数展開

文字列値に `${VAR_NAME}` または `${VAR_NAME:-default_value}` の記法を使うと、アプリ起動時に環境変数へ展開される。この仕様はClaude Codeの `mcp_config.json` と同じ記法に従う。

```json
{
  "anthropic_api_key": "${ANTHROPIC_API_KEY}",
  "some_url": "${API_BASE_URL:-https://api.example.com}/path"
}
```

- デフォルト値は `:-` の後に記載（省略可）
- 展開はアプリ起動時に `config.py` が一括処理する
- `mcp_config.json` の `env` フィールドおよび `args` フィールドも同様に展開する

---

## エラーハンドリング

| ケース | 挙動 |
|---|---|
| MCPツール失敗（リトライ上限） | ToolMessageに失敗情報を含め、orchestratorがコンテキストをもとに回答生成 |
| LLM API失敗（リトライ上限） | Slackに固定エラーメッセージを送信 |
| HTTP 4xx エラー | リトライしない（即時失敗扱い） |
| HTTP 5xx エラー | exponential backoffでリトライ |
| recursion_limit超過 | Slackに固定エラーメッセージを送信 |
| 全ツール失敗 | orchestratorが失敗を踏まえた上でメッセージを生成 |

---

## ファイル構成

```
slack_agent/
├── graph.py          # グラフ定義・エッジ・コンパイル
├── nodes.py          # orchestrator / tool_executor / compressor
├── state.py          # AgentState / CacheReference / CompressorResult
├── tools.py          # MCPツール定義・cache_fetcherビルトインツール
├── cache.py          # キャッシュストア実装（抽象クラス + インメモリ実装）
├── config.py         # モデル名・しきい値等の設定
├── retry.py          # exponential backoffリトライユーティリティ
└── slack_handler.py  # Slack Bolt連携・通知モジュール（Socket Mode）
settings.json         # アプリケーション設定
mcp_config.json       # MCPサーバー設定
```

---

## デプロイ

- Dockerコンテナとして実装
- 最終的にAWS ECSへデプロイ
- 環境変数・シークレットは `settings.json` で管理（ECS Task Definitionのsecretsと組み合わせることを想定）
- workerは1プロセス（複数Slack workspace対応は不要）

---

## 未決事項

- `recursion_limit` の上限値（現在のデフォルト案: 25）
- MCPツールの具体的なサーバー構成
- ECSタスクサイズ・スケーリング設定
