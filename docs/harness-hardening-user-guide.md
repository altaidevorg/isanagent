# Harness Hardening & Resilience User Guide

This guide details the hardening, isolation, and resilience features built into `isanagent` v0.13.0+, aligning with modern autonomous agent architectures and production requirements.

---

## 1. Secret-Safe Subprocess Environments

### Overview
By default, child processes spawned by agent tool execution (`exec`, `execution_run`, `python_run`, and UV environment invocations) **do not inherit master process host environment variables**.

This prevents accidental credential leakage into bash scripts, compiled binaries, test runners, or shell pipelines executed during autonomous agent sessions.

### Sanitization Policy (`ExecutionEnvironmentPolicy`)
- **Scrubbed Patterns**: Any environment variable matching sensitive substrings or prefixes is stripped:
  - `*_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, `*_AUTH`, `*_CREDENTIAL`
  - `OPENAI_*`, `ANTHROPIC_*`, `GEMINI_*`, `DEEPSEEK_*`, `GROQ_*`, `MISTRAL_*`, `COHERE_*`
  - `GITHUB_TOKEN`, `GH_TOKEN`, `SLACK_*`, `ALTAI_*`, `ISANAGENT_*`, `SSH_PASSWORD`
- **Preserved Safe Variables**: Platform-safe operating system variables are retained:
  - POSIX: `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`, `TMPDIR`, `TZ`, `PWD`
  - Windows: `SYSTEMROOT`, `WINDIR`, `COMSPEC`, `PATHEXT`, `TEMP`, `TMP`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `PROGRAMFILES`

### Explicit Variable Grants
If a specific task or workflow requires explicit environment variables in child processes:
```rust
use isanagent::environment::ExecutionEnvironmentPolicy;

let policy = ExecutionEnvironmentPolicy::default_safe()
    .with_allowlist(["CUSTOM_API_HOST", "RUN_BENCHMARK_FLAG"])
    .with_explicit_env(custom_env_map);
```

---

## 2. Metadata-Driven Tool Scheduling & Concurrency

### Overview
Every tool implements a policy contract classifying its execution mode:

```rust
use isanagent::traits::{ExecutionMode, ToolPolicy};

pub enum ExecutionMode {
    /// Read-only tool safe for concurrent execution (e.g. read_file, search_text, glob_files, list_dir).
    Parallel,
    /// Standard mutating tool requiring serial execution (e.g. write_file, edit_file).
    Serial,
    /// Mutates workspace or environment root; blocks until all pending tools finish (e.g. git_worktree).
    Barrier,
    /// Long-running asynchronous process (e.g. execution_run_background).
    Background,
}
```

### Built-in Tool Classification

| Tool | Policy Mode | Description |
| :--- | :--- | :--- |
| `read_file`, `glob_files`, `list_dir`, `search_text`, `web_search`, `web_fetch`, `search_memory`, `fetch_memory_by_date`, `search_tools`, `load_skill_instructions`, `arxiv_search`, `arxiv_fetch`, `hf_hub_file_fetch`, `execution_env_info`, `task_history_list` | **Parallel** | Read-only operations declared via each tool's typed `policy()` metadata (single source of truth — audit X1). The three ML research tools additionally require top-level `ml_domain_enabled = true` (opt-in, audit X4) and are absent from the registry by default. |
| All other built-in and MCP tools (`write_file`, `edit_file`, `exec`, …) | **Serial** (default) | Mutating or stateful tools executed in strict sequential order. |
| `git_worktree` | **Barrier** | Modifies working branches and root directories; drains all active tools before and after execution. |

---

## 3. Tool Result Cache (`tool_result_cache`)

### Overview
When tools produce massive outputs (large compiler traces, test suite dumps, dataset previews), dumping megabytes into the LLM conversation context causes context window saturation and reasoning degradation.

When auto-compaction swaps an oversized tool result out of the active conversation, the untruncated content is preserved in the session database (`agent_memory.db`, `tool_result_cache` table) and the message is replaced by an archival placeholder pointing at the original `tool_call_id`.

### Recovery with `recall_tool_result`
The agent can re-materialize the original content on demand:
```json
{
  "tool_call_id": "call_abc123"
}
```

Retention: the newest 500 cached results are kept per database (older rows pruned automatically).

---

## 4. Authoritative Session Projections

### Overview
Frontend applications (e.g. `altai-app`, ACP clients, Desktop GUIs) often need to display live task progress, todos, and background jobs. Previously, clients scraped tool calls or maintained divergent local stores.

`SessionProjection` provides server-owned, authoritative state snapshots emitted over the actor bus:

```rust
use isanagent::projections::SessionProjection;

let snapshot = SessionProjection::new("chat-123", seq, "running")
    .with_todos(active_todos)
    .with_subagents(running_subagents)
    .with_jobs(active_background_jobs);

// Emitted via BusMessage::SessionProjection(snapshot)
```

Clients consume `BusMessage::SessionProjection` as the single source of truth for UI state.

---

## 5. Native Provider Streaming Contract

### Overview
The `Provider` trait exposes native incremental chunk streaming:

```rust
use isanagent::traits::{Provider, StreamChunk, FinishReason};

#[async_trait]
pub trait Provider: Send + Sync + dyn_clone::DynClone {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<Value>,
    ) -> Result<LLMResponse, LLMError>;

    async fn stream(
        &self,
        messages: &[ChatMessage],
        tools: Option<Value>,
        sink: tokio::sync::mpsc::Sender<StreamChunk>,
    ) -> Result<LLMResponse, LLMError>;
}
```

### Emitted `StreamChunk` Variants
- `StreamChunk::TextDelta(String)`: Incremental completion text.
- `StreamChunk::ReasoningDelta(String)`: Thinking token streams.
- `StreamChunk::ToolCallDelta { id, name, args_delta }`: Progressive tool call streaming.
- `StreamChunk::Usage(TokenUsage)`: Real-time token accounting.
- `StreamChunk::Finish(FinishReason)`: Generation completion event.
