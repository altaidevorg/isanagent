# Future AI Development Blueprint (isanagent)

This document serves as the primary system architectural context for any future LLMs continuing to evolve the `isanagent` framework. If you are an AI tasked with writing a new tool, skill, integration, or feature for `isanagent`, please read this carefully to avoid producing anti-patterns or breaking the Actor Memory Model.

## 🧠 System Architecture Primer

`isanagent` completely decouples standard AI sequential blocking loops into a natively concurrent **Actor System**. 
The core data structure traveling the entire network natively is `isanagent::bus::BusMessage`:

```rust
pub enum BusMessage {
    Inbound(InboundMessage),
    Outbound(OutboundMessage),
    Telemetry(TelemetryEvent),
}
```

The fundamental philosophy here is **Wait-Free Threading**. Do not use `std::sync::Arc<std::sync::Mutex<rusqlite::Connection>>`. All critical I/O, especially database storage, must route through an opaque lock-free asynchronous Actor (`MemoryActor` wrapping `SqliteMemory` for instance).

### Workspace & Sandboxing
The agent is explicitly designed to run inside a sandbox. `IsanagentWorkspace` manages two boundaries:
1. `workspace_dir` (Outer Rim): Holds `config.toml`, generated logs, and `.system_generated` internal sqlite caches.
2. `sandbox_dir` (Inner Rim): This is the designated execution field, usually `workspace_dir/.agents`. This is where the Agent expects to load `AGENTS.md` and read `skills/`.

If you are writing a new `Tool` that mutates the disk or reads files, **DO NOT** let the Agent pass absolute system paths cleanly. You MUST wrap the injection using `crate::utils::resolve_path(&sandbox_dir, &agent_path`). Doing so naturally bounds all agent `../` directory escapes to the sandbox boundary.

Harness todo lists (`todo_write`) are stored in the workspace SQLite DB (same file as session memory: `<workspace_dir>/.system_generated/agent_memory.db`, table `harness_todos`). Reads and writes go through `MemoryMessage::ReplaceHarnessTodos` and `MemoryMessage::LoadHarnessTodos` on the same `SqliteMemoryActor` as session memory—never a separate mutex-wrapped connection. A legacy `<workspace_dir>/todos/*.json` folder from older builds is imported once when the memory actor opens the DB, then those files are removed.

User clarification (`ask_user`) sends an outbound message tagged with metadata `isanagent_clarification` and blocks the tool until the **next inbound** on the same session (`channel`, `chat_id`, optional `thread_id`). The agent routes that inbound to the waiting tool instead of starting a new reasoning task, so the model receives the reply as the tool result and continues the same turn.

Git worktrees (`git_worktree`): **off by default**. Set `[harness.git_worktree] enabled = true` in `config.toml` to register the tool. Worktree paths use the same `resolve_path` sandbox rules as other filesystem tools unless `allow_path_outside_sandbox = true`, which permits canonical paths outside the sandbox (for example a host temp directory). See `docs/harness-implementation-plan.md` Phase 4.

Sub-agents (`subagent_spawn`, `task_*`, `subagent_plan_execute`): **off by default** via `[harness.subagents] enabled = true`. Sub-agents run a second `run_reasoning_loop` with a synthetic chat id (`subagent-…`) and optional tool allowlist. `cancel_children_on_parent_cancel` (default true) controls whether cancelling or superseding the parent chat’s reasoning also cancels those child tasks. See `docs/harness-implementation-plan.md` Phase 5.

### Structured LLM Extraction
If you are asking the LLM to yield a structured JSON payload internally (e.g. for reflection or summarization outside of the standard `ToolCall` registry):
**DO NOT** use brittle string matching like `text.find('{')`. 
**DO USE** `crate::utils::extract_json_from_llm_response(&text)` to safely isolate the payload from conversational wrappers or markdown blocks.

## 🛠 Adding a newly Native Rust Tool

Tools act as the fundamental abilities the Agent uses via JSON schema during its core sequential processing loop inside `AgentLogic`.

1. **Implement `Tool`**: Create a struct in `src/tools/builtin.rs` (filesystem / web / shell) or `src/tools/workflow.rs` (session todos, tool search, `ask_user`, etc.) and implement the `async_trait` `Tool`.
   - `name`: Strict string representing the tool call name.
   - `description`: The prompt instruction to the LLM on *how* to use it.
   - `input_schema`: Use `serde_json::json!` to define a rigid JSON schema the LLM must map arguments to.
   - `execute`: The async function resolving the payload. Returns `Result<String, String>` (Ok/Err).

2. **Register Tool**: In `src/bin/isanagent.rs`, inject the new initialized instance into the global `ToolRegistry` mapped to `AgentLogic`. 

### Proactive Networking during Tools
If your Tool is incredibly slow (Scraping a massive database, generating an image, compiling code), you should update the user in real-time. Do not await in silence. 
Inject the `tokio::sync::mpsc::Sender<BusMessage>` channel directly into your tool at creation:
```rust
let (tx, _rx) = tokio::sync::mpsc::channel(100);
let my_slow_tool = MySlowTool::new(tx.clone());

// Inside execute() loop:
tx.send(BusMessage::Outbound(OutboundMessage { ... })).await.unwrap();
```
*Note*: This multiplexed bus routes directly back to whatever channel (Terminal, Slack thread, API) triggered the execution perfectly via its matching `chat_id`. 

## 📝 Adding an AI Skill (Dynamic Markdown)

Don't write Rust code if the problem is strictly formatting, instructions, or contextual workflow. 

Write a `SKILL.md` inside `workspace/.agents/skills/{skill-name}/SKILL.md`.

```yaml
---
name: code_review
description: Analyze Rust lifetime loops
requires:
    bins: ["cargo", "rustc"]
    env: ["GITHUB_TOKEN"]
always: false
---

# Instructions
When checking code...
```

- When `always` is **false**, the Agent will see its `description` block inside the System Prompt. It will dynamically call a built-in standard tool `load_skill_instructions` if it wants to learn how to do the specific pipeline requested.
- When `always` is **true**, the raw markdown body is forcibly concatenated directly into every system prompt. This drastically eats context window but ensures rigid behavioral obedience (like global formatting rules).
- The `requires` object automatically hooks into `which` and `std::env` at startup. If the environment is missing a binary or token, the Agent actively sees it marked as `[❌ UNAVAILABLE MISSING GITHUB_TOKEN]` inside context to prevent hallucinated tool execution failures.

## 🏢 Implementing a new Channel

`isanagent` has distinct platforms (Channels) polling concurrently. E.g., `TerminalChannel`, `SlackChannel`, `EmailChannel`. 

1. **Implement `Channel`**: Define your struct in `src/channels/{platform}.rs`.
   - It needs to run a detached background `tokio` thread checking its respective networking protocol endpoint forever.
   - On inbound parsing, construct `InboundMessage` matching your external packet origin (`chat_id` typically maps to UUIDs, Slack Rooms, or EMail IMAP keys). Send this to the core Actor Bus.
   - In `Altbot`, you'll spawn this receiver thread explicitly.
2. **Handle Outbound**: You must also expose an asynchronous listener or method specifically catching `OutboundMessage` packets flowing out of the Actor Bus so you can map the plain string response back into your network's specific protocol (e.g. `slack.chat.postMessage()`).

## 📊 Telemetry Output

If you add a new metric or LLM analytics block, format it explicitly as a `BusMessage::Telemetry(TelemetryEvent)` payload. The `WorkspaceLoggingActor` natively captures, serializes, and writes these structured JSON traces reliably to `conversation.jsonl` acting as the sole analytical pipeline.

## 🧠 Memory & Reflection Pipelines

The memory system has two distinct phases operating under the Actor loop:
1. **Active Auto-Compaction** (`src/agent/mod.rs`): If an active chat exceeds token/turn limits, a blocking summarization request is made, saved out to `SqliteMemoryActor` safely, and the raw history buffers are cleared to protect context limits.
2. **Idle Background Reflection** (`src/reflection.rs`): An asynchronous supervisor checks idle sessions on an interval. When a predefined threshold of SQLite summaries is reached, it spawns a task converting them into a single `MEMORY.md` file saved physically to the `workspace_dir`.


## Development workflow
After implementing a feature, follow this exact workflow to deliver high-quality code.

1. Review your code for performance-oriented move semantics, graceful handling of errors and options, and correct async usage. As a rule of thumb, avoid `.unwrap()` and `.expect()` that can cause panics at runtime.
2. Run `cargo clippy` and keep your clippy happy --it's your best friend. No Allow() macro allowed to suppress clippy warnings.
3. Run `cargo fmt` so that it's always well-formatted, avoiding unnecessary diffs that simply come from formatting.
4. This is a living document --keep this document up-to-date as you introduce new features and/or architectures.

## Note for Windows
On windows, building the project in debug mode causes a PDB-related linker error. Build the project in release mode on Windows instead.
