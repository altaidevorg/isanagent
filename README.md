# Agent-RS

Agent-RS is a high-performance, actor-based framework for building complex AI Agent pipelines and unified digital workspaces in Rust. Instead of simple single-threaded request/response scripts, Agent-RS leverages the **Actor Model** to distribute agent logic across parallel, supervisor-tethered nodes, enabling persistent asynchronous messaging over terminal interfaces, Slack bots, and Email.

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

Agent-RS was explicitly built for robust multi-modal Artificial Intelligence. Instead of the Agent being frozen inside a blocked blocking `await` loop when dealing with tools, the Agent itself runs as a decoupled Actor receiving a stream of messages from various user networks. 

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

### 1. Build The Project 
```bash
cargo build --release
```

### 2. Configure Your Workspace
Agent-RS expects a "Workspace" root containing your configuration and where the Agent is allowed to read/write state securely.

Create a directory (e.g., `my_agent`) and place a `config.toml`:
```toml
[app]
max_iterations = 25
max_tool_output_chars = 3000
restrict_to_workspace = true

[provider]
model_name = "gemini-2.5-flash"
api_key_env = "GEMINI_API_KEY"

[slack]
enabled = false
app_token = "xapp-..."
bot_token = "xoxb-..."
reply_in_thread = true
reaction_emoji = "eyes"

[memory]
enabled = true
short_term_threshold_turns = 20
short_term_threshold_tokens = 100000
short_term_threshold_mins = 15
long_term_interval_mins = 360
max_recent_summaries = 5
long_term_threshold_summaries = 5
```

### 3. Run the Agent
```bash
# Pass your workspace path (and optionally, config relative to the binary).
cargo run --bin altbot -- --workspace my_agent --config my_agent/config.toml
```

## � Configuring Channels

The `[channel_name]` blocks in `config.toml` allow you to bring external integrations online cleanly:

### HTTP API Channel
Start a local REST server compatible with OpenAI JSON schemas for testing the core engine programmatically. 
```toml
[api]
enabled = true
port = 8080
```
```bash
curl http://localhost:8080/v1/chat/completions -d '{"message": "Hello!"}'
```

### Slack Socket Mode
Listens directly to Slack Channels without requiring an ingress webhook proxy. Includes built-in exponential backoff for Socket drops, outbound message retry logic, and configurable immediate Emoji processing acknowledgments.
```toml
[slack]
enabled = true
app_token = "xapp-..."  # Requires Socket Mode toggled in Slack App Settings
bot_token = "xoxb-..."
reply_in_thread = true
reaction_emoji = "eyes" # Emoji assigned immediately to user messages signaling receipt
```

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
Agent-RS automatically compacts conversational history into structured SQLite JSON summaries when session length thresholds (`short_term_threshold_turns` or `short_term_threshold_tokens`) are exceeded. 
When enough short-term summaries accumulate, a background reflection engine consolidates the information into a long-term `MEMORY.md` injected directly into the active Agent workspace context.

## 🛠 Tools and Skills

Agent-RS supports dual-layer extensibility: Built-in strict Rust Tools, and dynamic `Markdown` LLM Skills.

### Built-in Tools
Tools like `web_scrape`, `shell_exec`, `list_dir`, `read_file` are mapped natively in `src/tools/builtin.rs`. If `restrict_to_workspace` is true, Agent-RS heavily validates file-system calls preventing the AI from path traversing outside the active workspace directory or manipulating system state.

### Markdown Skills (`/workspace/.agents/skills/`)
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
For specific guidance on developing new Tools, Skills, or contributing to the architecture as an automated agent yourself, please refer to the dedicated [`GEMINI.md`](./GEMINI.md) blueprint document.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
