# isanagent

isanagent is a high-performance, actor-based framework for building complex AI Agent pipelines and unified digital workspaces in Rust. Instead of simple single-threaded request/response scripts, isanagent leverages the **Actor Model** to distribute agent logic across parallel, supervisor-tethered nodes, enabling persistent asynchronous messaging over terminal interfaces, Slack bots, and Email.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [The Agent Architecture](#the-agent-architecture)
- [Quick Start](#quick-start)
- [Configuring Channels](#configuring-channels)
- [Tools and Skills](#tools-and-skills)
- [Development Guide](#development-guide)

## 🔭 Overview

isanagent was explicitly built for robust multi-modal Artificial Intelligence. Instead of the Agent being frozen inside a blocked blocking `await` loop when dealing with tools, the Agent itself runs as a decoupled Actor receiving a stream of messages from various user networks.

**Why an Actor Model?**
- **100% Lock-Free Concurrency**: SQLite Memory channels and Context Assembly run without centralized `Arc<Mutex>` locks, avoiding thread contention natively.
- **Asynchronous Channels**: The agent can process incoming emails natively in the background without affecting a real-time terminal or Slack session.
- **Resilience**: Supervisors wrap critical tasks (like disk IO logging) and gracefully restart crashed nodes on fatal panics.

## 🌟 Key Features

- **Multi-Channel Multiplexing**: Native channels for **Terminal (CLI)**, **Slack Socket Mode**, **IMAP/SMTP Email**, and a local **HTTP API**.
- **Isolated SQLite Worker Memory**: Seamless Context window persistence per channel thread. (E.g., talking to the agent on Slack `D1234` is distinct memory from talking to the agent on Terminal).
- **Proactive Multi-Step Reasoning**: Agents can yield dynamic intermediate `BusMessage::Outbound` packets to the user during long multi-tool execution sequences.
- **Telemetry & Observability**: Granular log streaming (capturing `reasoning_content`, `prompt_tokens`, `ToolCall`, etc.) output natively to `.system_generated/logs/conversation.jsonl`.
- **Progressive Skill Loading**: Exposes complex Anthropic-style YAML Markdown files directly into the LLM context only when the Agent requests the tool via dependency-validated schema registries.

## 🧠 The Agent Architecture

The core message bus routes information between interfaces (e.g. Slack, Terminal) and the LLM execution logic.

1.  **Channels**: Poll external networks or standard input and emit `BusMessage::Inbound(InboundMessage)` tagged with a distinct `chat_id` and `channel` origin.
2.  **Session Manager & Memory**: The Actor receives the Inbound envelope, delegates context fetching to an internal Lock-Free SQLite Memory Actor by hashing `channel:chat_id:thread_id`, and injects the previous message buffer alongside the new prompt.
3.  **Tool Execution**: The Agent evaluates the prompt and emits `ToolCall` requests. Built-in tools process (with strict workspace dir-sandboxing and bounded execution time) and return results without holding up the global bus.
4.  **Multiplexed Output**: Once the run completes, the Agent fires a `BusMessage::Outbound(OutboundMessage)` back to the central bus router, which pipes the envelope explicitly to the original Channel that requested it.

## 🚀 Quick Start

### 1. Build The Embedded UI
```bash
cd ui
npm ci
npm run build
cd ..
```

isanagent embeds `ui/dist` into the Rust binary at compile time. If `ui/dist/index.html` is missing,
`cargo build` / `cargo run` will fail with an actionable error.

### 2. Build The Project
```bash
cargo build --release
```

### 3. Bootstrap Your Workspace
isanagent now ships with an onboarding command that creates a ready-to-edit workspace root, starter config, sandbox directory, and local skills.

```bash
cargo run --bin isanagent -- onboard --workspace my_agent
```

This creates:

```text
my_agent/
├── config.toml
├── .system_generated/
└── workspace/
    ├── AGENTS.md
    ├── USER.md
    ├── SOUL.md
    └── skills/
        ├── cron/
        │   └── SKILL.md
        └── skill-creator/
            └── SKILL.md
```

The generated `config.toml` starts with these defaults:
```toml
restrict_to_workspace = true
max_iterations = 50
max_tool_output_chars = 10000
# max_web_tool_output_chars = 50000  # optional; caps web_search / web_fetch (default 50000)

[provider]
model_name = "gemini-3-flash-preview"
api_key_env = "GEMINI_API_KEY"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

[slack]
enabled = true
mode = "webhook"
bot_token = "<changethis>"
signing_secret = "<changethis>"
webhook_port = 8090
webhook_path = "/slack/events"
reply_in_thread = true
reaction_emoji = "eyes"
# Socket mode example:
# mode = "socket"
# app_token = "<changethis>"

[email]
enabled = false
imap_host = "imap.agentmail.to"
imap_port = 993
imap_username = "<changethis>"
imap_password = "<changethis>"
smtp_host = "smtp.agentmail.to"
smtp_port = 465
email_address = "<changethis>"

[api]
enabled = false
port = 8080
serve_ui = false
# bind_address = "127.0.0.1" # defaults to 127.0.0.1 when serve_ui = true, else 0.0.0.0

[multi_tenant_edge]
activity_heartbeat_enabled = false
cron_scheduling_enabled = false

[memory]
enabled = true
short_term_threshold_turns = 20
short_term_threshold_tokens = 100000
short_term_threshold_mins = 15
long_term_interval_mins = 360
long_term_threshold_summaries = 5
max_recent_summaries = 5
```

Update the `<changethis>` placeholders, ensure `GEMINI_API_KEY` is set, and disable any channels you are not ready to use yet before running the agent.
For example, if you are not using Slack on first boot, set `[slack].enabled = false` in `config.toml`.

### 4. Run the Agent
```bash
# Pass your workspace path. `config.toml` defaults to <workspace>/config.toml.
cargo run --bin isanagent -- --workspace my_agent
```

## Configuring Channels

The `[channel_name]` blocks in `config.toml` allow you to bring external integrations online cleanly:

### HTTP API Channel
Start a local REST server compatible with OpenAI JSON schemas for testing the core engine programmatically.
```toml
[api]
enabled = true
port = 8080
serve_ui = false
# bind_address = "0.0.0.0"
```
```bash
curl http://localhost:8080/v1/chat/completions -d '{"message": "Hello!"}'
```

The API channel also supports a stateful `responses` flow:

```bash
curl http://localhost:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello!"}'
```

The response includes an `id` that can be used as `previous_response_id` on the next request:

```bash
curl http://localhost:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"Continue the previous conversation.","previous_response_id":"resp_..."}'
```

Notes:
- `store` defaults to `true` when omitted, so responses are persisted and can be continued later.
- Set `"store": false` if you do not want a response persisted for later `previous_response_id` lookups.
- `responses` accepts text-oriented JSON input and normalizes it into a single message before sending it into the agent runtime.

### Built-in UI
Enable the built-in browser UI on the same API listener. The UI calls the existing `/v1/responses`
endpoint same-origin and keeps conversation state in the browser.

```toml
[api]
enabled = true
port = 8080
serve_ui = true
bind_address = "127.0.0.1"
```

Then open `http://127.0.0.1:8080/`.

Notes:
- `bind_address` is optional. When omitted, it defaults to `127.0.0.1` if `serve_ui = true`, otherwise `0.0.0.0`.
- The UI does not add a new chat API surface. It uses the existing `responses` flow directly.
- Browser-local persistence means conversations continue across refreshes in the same browser, but not across devices.

### Multi-Tenant Edge Heartbeats
When isanagent runs behind `multi-tenant-edge`, it can keep the instance alive during long background tool execution by heartbeating `POST /_internal/activity`.

```toml
[multi_tenant_edge]
activity_heartbeat_enabled = true
```

When enabled, isanagent reads `MTE_PROXY_BASE_URL`, `MTE_ACTIVITY_SECRET`, and `MTE_HEARTBEAT_TTL_MS` from the process environment and sends immediate + periodic heartbeats only while tools are running. If those env vars are missing or the heartbeat endpoint rejects the call, isanagent logs a warning and continues the tool run without failing the request.

### Multi-Tenant Edge Cron Scheduling
When isanagent runs behind `multi-tenant-edge`, it can delegate cron scheduling to `PUT /_internal/crons` so the instance can sleep between jobs and wake back up through cron webhooks.

```toml
[api]
enabled = true
port = 8080

[multi_tenant_edge]
cron_scheduling_enabled = true
```

When enabled, isanagent:
- Requires `[api].enabled = true` because the edge wakes jobs through `GET /_mte/cron/:job_id/:token` on the API listener.
- Reads `MTE_PROXY_BASE_URL` and `MTE_CRON_SECRET` from the process environment and fails fast if they are missing or invalid.
- Keeps storing cron jobs locally in SQLite, but pushes the full rule set to `multi-tenant-edge` on startup and after add/remove changes.
- Supports `cron_expr` only as a 6-field UTC cron expression (`second minute hour day month day-of-week`) and supports one-shot `at` schedules by converting them into one edge cron rule.
- Rejects `every_seconds` while MTE cron scheduling is enabled.

### Slack Modes
Slack now supports both Socket Mode and a dedicated webhook listener for Slack Events API. Webhook mode runs on its own HTTP listener and does not share the API channel port.

#### Slack Socket Mode
```toml
[slack]
enabled = true
mode = "socket"
app_token = "xapp-..."  # Requires Socket Mode toggled in Slack App Settings
bot_token = "xoxb-..."
reply_in_thread = true
reaction_emoji = "eyes" # Emoji assigned immediately to user messages signaling receipt
```

#### Slack Webhook Mode
```toml
[slack]
enabled = true
mode = "webhook"
bot_token = "xoxb-..."
signing_secret = "..."
webhook_port = 8090
webhook_path = "/slack/events" # Optional, defaults to /slack/events
reply_in_thread = true
reaction_emoji = "eyes"
```

Notes:
- Webhook mode listens on a dedicated port and routes incoming Slack Events API payloads into the same agent flow as Socket Mode.
- This mode does not yet include OAuth installation flow, tenant token routing, or a shared ingress layer with the API channel.

### Email Pipeline
Uses IMAP `Idler` threads and an SMTP transport pool.
```toml
[email]
enabled = true
imap_host = "imap.example.com"
imap_username = "bot@example.com"
imap_password = "..." # Recommened: Set via Env Var override
smtp_host = "smtp.example.com" # etc
```

### Advanced Memory & Auto-Compaction
isanagent automatically compacts conversational history into structured SQLite JSON summaries when session length thresholds (`short_term_threshold_turns` or `short_term_threshold_tokens`) are exceeded.
When enough short-term summaries accumulate, a background reflection engine consolidates the information into a long-term `MEMORY.md` injected directly into the active Agent workspace context.

## 🛠 Tools and Skills

isanagent supports dual-layer extensibility: Built-in strict Rust Tools, and dynamic `Markdown` LLM Skills.

### Built-in Tools
Tools like `web_search`, `web_fetch`, `exec` (shell), `list_dir`, `read_file`, `write_file`, `edit_file`, `glob_files`, and `search_text` are implemented in `src/tools/builtin.rs`. Workflow tools (`todo_write`, `search_tools`, `ask_user`, …) live in `src/tools/workflow.rs`. If `restrict_to_workspace` is true, filesystem tools validate paths so the model cannot traverse outside the workspace sandbox.

Optional **harness** features are config-gated under `[harness.*]` in `config.toml` (for example `git_worktree` and `subagents`). Behaviour, acceptance checks, and Phase 6 roadmap are documented in [`docs/harness-implementation-plan.md`](./docs/harness-implementation-plan.md). Architecture notes for tools, memory, and sandboxing are in [`AGENTS.md`](./AGENTS.md).

### Markdown Skills (`/workspace/skills/`)
Skills provide complex workflows, templates, or instructions natively to the Agent without recompiling Rust code.
Place a directory in `skills/` containing a `SKILL.md` file using YAML frontmatter:

```markdown
---
name: create_dockerfile
description: Write optimal Rust Alpine dockerfiles
requires:
  bins: ["docker"]
always: false
---

# Docker Instructions
When asked to containerize this app, you MUST use cargo-chef and multi-stage builds...
```

The Agent will see the capability in its system prompt and can dynamically call `load_skill_instructions(name: "create_dockerfile")` to inject this explicitly when needed.

## 🤝 Development Guide
For specific guidance on developing new Tools, Skills, or contributing to the architecture as an automated agent yourself, please refer to the dedicated [`AGENTS.md`](./AGENTS.md) blueprint document and [`AGENTS.md`](./AGENTS.md).

Before opening a PR, from the repo root:

```bash
cargo fmt
cargo clippy --release -p isanagent --all-targets
cargo test --release -p isanagent
```

On Windows, prefer `--release` for builds and tests if you hit PDB linker issues in debug mode (see `AGENTS.md`).

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
