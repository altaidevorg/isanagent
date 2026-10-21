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

---

## 6. Tool Permissions (`[harness.permissions]`)

### Overview
Every tool call passes a permission check before it runs. File tools are judged by **where** the
path they touch lives; other tools by a default or a per-tool entry. Each rule is `allow`, `ask`, or
`deny`.

```toml
[harness.permissions]
inside_workspace = "allow"   # file tools on paths inside the sandbox
outside_workspace = "ask"    # the same tools on paths outside the sandbox
default = "allow"            # every other tool not covered below or by the shell policy

[harness.permissions.tools]  # per-tool override by exact tool name; wins over the rules above
# web_fetch = "ask"
# git_worktree = "deny"
# mcp__github__create_issue = "ask"   # MCP-proxied and plugin tools use their registered name
# exec = "ask"                        # replaces [harness.shell_policy].mode for exec (interactive only)
```

| Key | Default | Applies to |
| :--- | :--- | :--- |
| `inside_workspace` | `allow` | `read_file`, `write_file`, `edit_file`, `list_dir`, `glob_files`, `search_text`, `exec` (working directory), `git_worktree` on paths inside `sandbox_dir` |
| `outside_workspace` | `ask` | the same tools on any other path |
| `default` | `allow` | web, memory, cron, MCP, plugin and other tools without paths (`ask_user` is never gated) |
| `tools.<name>` | — | that tool only; replaces the location and default rules for it (for `exec`, `exec_send`, `execution_run*`: see *Shell policy* below) |

### Inside vs. outside
- **Inside** means inside `sandbox_dir` (`workspace_dir/workspace` by default, or the `sandbox` /
  host project directory). The outer `workspace_dir` — `config.toml`, `.system_generated/` (session
  database, logs, checkpoints) — is **outside**, so the agent cannot read or change its own config
  and secrets without your approval.
- Paths are resolved exactly as the tool will resolve them: canonicalized, symlinks followed, and a
  file that does not exist yet is anchored on its nearest existing ancestor. `../` escapes and
  symlinks that lead out of the sandbox count as **outside**; a dangling symlink is judged by its
  target (writing through it would create the target).
- A call that touches several paths (e.g. `git_worktree add`) is judged by its strictest path.
- A file tool that reports no path for a call is judged by the stricter of `inside_workspace` and
  `outside_workspace`, never by `default`.

### Approval prompts
`ask` uses the same approval channel as shell and edit approvals (web approval card, TUI prompt,
`ask_user` over the API) with the choices **approve**, **deny**, **always**, **abort**:
- File reads / listings / working directories show the tool, the operation and the **canonical
  absolute path**, plus what **always** will cover (e.g. "always = allow reads under `/data`
  until isanagent restarts").
- Edits (`write_file` / `edit_file`) show the diff as before, with the absolute path and a note
  when the file is outside the sandbox.
- **always on an outside path** grants the same operation kind — reads, writes/edits, or running
  commands (working directory), kept separate — for the target's **directory and everything below
  it** until the process restarts: the parent folder for a file, the folder itself for a
  directory target (`list_dir`, `glob_files`, `search_text`, `exec` working directory). Matching is
  per path component (`/data2` is not under `/data`) and ignores case on macOS and Windows.
  A grant never widens onto the outer workspace's `config.toml` or `.system_generated/` (so
  "always" on `~/x` does not cover `~/.isanagent/config.toml`) unless the granted folder itself
  lies inside them (the prompt then says "… (except isanagent config and data)"); the exact path you
  approved is always covered. If the folder would be a filesystem root (`/`, `C:\`), your home
  directory itself, or an ancestor of the outer workspace folder, **always** falls back to the exact
  path ("always = allow reads of this file only"). Inside the sandbox (when
  `inside_workspace = "ask"`) and for legacy `edit_mode = "ask"` prompts, **always** covers that
  exact path only.
- Other tools show their JSON arguments; **always** allows the same tool with the same arguments.
- **deny** returns a policy error to the model; **abort** stops the run.

### Unattended sessions
Nobody can answer a prompt in an autonomous / headless session (`isanagent_autonomous_session`), so
every `ask` resolves to `deny` there — the same rule the shell policy uses. `allow` stays `allow`:
with the defaults an autonomous run can read and edit inside the sandbox but not outside it.

### Precedence and legacy keys
1. `[harness.permissions.tools].<name>` (most specific).
2. `[harness.permissions]` keys you set explicitly.
3. Legacy keys: `restrict_to_workspace = true` → `outside_workspace = "ask"` (it no longer means the
   hard sandbox: `..` is not clamped at the sandbox root any more, it resolves to the real parent
   and asks); `= false` → `outside_workspace = "allow"`. Set `outside_workspace = "deny"` for the old
   hard sandbox.
4. Defaults: inside `allow`, outside `ask`, default `allow`.

`[harness.shell_policy].edit_mode` and `edit_unattended_default`, when set, are a **ceiling** for
file edits: the stricter of them and the rule above wins, even over a per-tool `allow`. This keeps
plan mode (`edit_mode = "deny"`) blocking every edit, and `edit_mode = "ask"` restores a prompt for
edits inside the sandbox. Left unset (the default), edits follow `[harness.permissions]`.

The shell policy (`[harness.shell_policy]`: `mode`, risky-pattern approvals, `unattended_default`)
governs what `exec`, `exec_send` and `execution_*` run. A `[harness.permissions.tools]` entry for one
of these tools does **not** replace the shell policy; it replaces only `[harness.shell_policy].mode`
for that tool in interactive sessions:
- `allow`: no prompts, as with `mode = "allow"`.
- `ask`: the shell-policy prompt for risky commands (approval patterns; every `execution_run*` call),
  as with `mode = "ask"`. Safe `exec` commands run without a prompt.
- `deny`: the tool is off — every call is blocked.

Unattended sessions use the **stricter** of the entry (`ask` read as `deny`) and
`unattended_default`, so an entry can never loosen autonomous runs: with the default
`unattended_default = "deny"`, `exec = "allow"` still blocks risky commands there. Approval prompts
and `ShellPolicyDecision` telemetry are the shell policy's own, and the `exec` working directory is
still judged by `inside_workspace` / `outside_workspace`. Steering hooks run first and can only
tighten (`permissionDecision: "ask"` turns `allow` into a prompt; it never lifts a `deny`).

### Hard sandbox and other boundaries
- A tool that can never reach outside the sandbox (outside resolves to `deny` interactively and
  unattended, i.e. an explicit `outside_workspace = "deny"` or a per-tool `deny`) keeps the hard
  `resolve_path` boundary: outside paths fail and `..` is clamped at the sandbox root.
- `execution_*` working directories are resolved by the execution provider, so they leave the
  sandbox only when outside access resolves to `allow` in both interactive and unattended sessions
  (`outside_workspace = "allow"`; a per-tool `execution_run` entry does not change this).
- `checkpoint` restores stay confined to the sandbox unless outside edits are `allow` in both
  interactive and unattended sessions (an `edit_unattended_default` ceiling keeps them confined).
- `[harness.git_worktree] allow_path_outside_sandbox = true` remains an explicit grant: worktree
  paths are then not subject to the location rules.
- Embedding hosts: `HostPermissionMode::Bypass` also sets `outside_workspace = "allow"`;
  `Plan`, `Ask`, `AutoEdit` keep mapping to `edit_mode` (`deny`, `ask`, `allow`).

### Settings screens
The web **Settings → Harness** tab and the TUI `/settings` harness screen edit
`inside_workspace` / `outside_workspace` (allow / ask / deny). Saving writes
`[harness.permissions]` and removes the legacy `restrict_to_workspace` key. `GET/PATCH
/v1/settings/harness` returns `inside_workspace`, `outside_workspace` and, for older clients,
`restrict_to_workspace` (`true` unless outside is `allow`); a PATCH with only the boolean is still
accepted (`true` → `ask`, `false` → `allow`). Changes apply after a restart.
