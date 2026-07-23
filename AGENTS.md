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
    Log(LogEvent),
    LoggerControl(LoggerControlMessage),
    Cancel(String), // chat_id — explicit stop; clears FIFO queued inbounds for that chat
}
```

While a **main** reasoning loop is already active for a `chat_id`, additional **`Inbound`** messages for that chat are held in a **per-chat FIFO** and started only after the current turn finishes. **`BusMessage::Cancel(chat_id)`** (for example terminal **`/cancel`** / **`/stop`**) is the explicit stop path: it cancels the in-flight loop, optionally cascades to sub-agents when configured, and **drops** any queued prompts for that chat. New user text does **not** implicitly cancel an in-flight turn.

The fundamental philosophy here is **Wait-Free Threading**. Do not use `std::sync::Arc<std::sync::Mutex<rusqlite::Connection>>`. All critical I/O, especially database storage, must route through an opaque lock-free asynchronous Actor (`MemoryActor` wrapping `SqliteMemory` for instance).

### Workspace & Sandboxing
The agent is explicitly designed to run inside a sandbox. `IsanagentWorkspace` manages two boundaries:
1. `workspace_dir` (Outer Rim): Holds `config.toml`, generated logs, and `.system_generated` internal sqlite caches.
2. `sandbox_dir` (Inner Rim): This is the designated execution field, usually `workspace_dir/.agents`. This is where the Agent expects to load `AGENTS.md` and read `skills/`.

If you are writing a new `Tool` that mutates the disk or reads files, **DO NOT** let the Agent pass absolute system paths cleanly. You MUST wrap the injection using `crate::utils::resolve_path(&sandbox_dir, &agent_path`). Doing so naturally bounds all agent `../` directory escapes to the sandbox boundary.

Harness todo lists (`todo_write`) are stored in the workspace SQLite DB (same file as session memory: `<workspace_dir>/.system_generated/agent_memory.db`, table `harness_todos`). Reads and writes go through `MemoryMessage::ReplaceHarnessTodos` and `MemoryMessage::LoadHarnessTodos` on the same `SqliteMemoryActor` as session memory—never a separate mutex-wrapped connection.

User clarification (`ask_user`) sends an outbound message tagged with metadata `isanagent_clarification` and blocks the tool until the **next inbound** on the same session (`channel`, `chat_id`, optional `thread_id`). The agent routes that inbound to the waiting tool instead of starting a new reasoning task, so the model receives the reply as the tool result and continues the same turn.

Git worktrees (`git_worktree`): **off** in a minimal `config.toml` until you set `[harness.git_worktree] enabled = true`. The onboarding template under `assets/onboarding/config.toml` turns this **on** for new workspaces. Worktree paths use the same `resolve_path` sandbox rules as other filesystem tools unless `allow_path_outside_sandbox = true`, which permits canonical paths outside the sandbox (for example a host temp directory). See `docs/harness-implementation-plan.md` Phase 4.

Sub-agents (`subagent_spawn`, `task_*`, `subagent_plan_execute`, `task_history_list`): **off** in a minimal config until `[harness.subagents] enabled = true`. The onboarding template enables this for new workspaces. Sub-agents run a second `run_reasoning_loop` with a synthetic chat id (`subagent-…`) and optional tool allowlist. `cancel_children_on_parent_cancel` (default true) controls whether an **explicit** parent cancel (`BusMessage::Cancel` / API cancel / terminal **`/cancel`**) also cancels those child tasks (not triggered by queued follow-up user messages). Completed runs are recorded in the workspace SQLite table **`subagent_tasks`** (same DB as session memory). See `docs/harness-implementation-plan.md` Phase 5.

**Named agents** (Phase 5b): `subagent_spawn` now accepts an optional `agent` parameter to invoke specialized agents defined in config. Four built-in defaults are registered when no `[agents.<name>]` blocks exist in `config.toml`: **Semble Scout** (modelsiz local code retrieval), **researcher** (web+read tools, temp 0.1), **coder** (full harness allowlist, temp 0.2), and **evaluator** (read-only review tools, temp 0.0). Semble Scout is a deterministic `mode = "semble_scout"` worker: it invokes `uvx --from semble[mcp]==0.5.1 semble search` against the host-supplied project root, never accepts a root from the LLM, and keeps its cache under `.isanagent/cache/semble`. Tools: `agent_list` (list available agents), `task_dashboard` (unified active+history view). The coordinator system prompt is auto-injected with agent descriptions. `wake_on_completion` (default true) enqueues a synthetic inbound when a subagent finishes so the parent can consume the result without polling. Config keys (under `[harness.subagents]`): `wake_on_completion`, `task_history_retention`. Agent definitions use `[agents.<name>]` blocks or `[harness.agents.<name>]` (harness-level wins in merge): `description` (required), `mode` (`"subagent"` or `"semble_scout"`), `system_prompt`, `system_prompt_file`, `allowed_tools`, `model`, `temperature`, `max_iterations`, `hidden`, `color`. Implementation: `src/agent/registry.rs`, `src/agent/subagent.rs`. **`subagent_plan_execute`** accepts optional per-step `"agent"` in plan JSON (same schema as spawn).

**MaxEvolve kernel porting** (`[harness.kernel_porting] enabled = true`): Triton/PyTorch→JAX/Pallas workflow via named agents (`kernel_orchestrator`, `gpu_to_jax`, `evolve_orchestrator`, …), `kernel-porting` skill, Python validators under `skills/kernel-porting/scripts/`, MAP-Elites tools (`kernel_db_init`, `kernel_db_sample`, `kernel_db_insert`, `kernel_db_status` in `src/tools/kernel_porting.rs`), and Colab/SSH hardware profiling. Operator guide: `docs/kernel-porting-user-guide.md`. Projects live under `kernels/projects/{id}/`. Onboarding copies prompts to `.agents/prompts/`, reference docs to `kernels/reference/`, and benchmarks to `benchmarks/`.

**AutoTrainess post-training** (`[harness.autotrainess] enabled = true`): autonomous LLM post-training ACI (plan→data→train→eval→log) via named agents (`train_orchestrator`, `train_planner`, `data_prep`, `trainer`, `train_evaluator`), `autotrainess` skill (adapted from AutoTrainHub), LlamaFactory scripts under `skills/autotrainess/aci/train/scripts/`, and experiment-ledger tools (`train_db_init`, `train_db_append`, `train_db_list`, `train_db_status`, `train_db_get` in `src/tools/autotrainess.rs`). Training runs through the existing execution plane (prefer SSH GPU). Operator guide: `docs/autotrainess-user-guide.md`. Projects live under `train/projects/{id}/`. Onboarding copies the skill tree and iteration plan to `.agents/autotrainess/`.

**ML engineer overlay** (`[harness.ml_engineer] enabled = true`): appends the embedded policy from `src/ml_engineer.rs` (source text in repo `assets/ml_engineer_overlay.md`). New workspaces get a **reference copy** at **`workspace/ML_ENGINEER_OVERLAY.md`** from `isanagent onboard` (editing it does not change runtime). Optional **`workspace/ML_POLICY.md`** is merged when present (`compile_system_prompt`). Inbound metadata **`isanagent_autonomous_forbid_final_without_tools`** (and optional **`isanagent_autonomous_until`**) tune autonomous-style sessions; config key **`forbid_final_without_tools`** sets a default nudge so the main loop can require tool use before ending a step.

**Shell policy** (`[harness.shell_policy]`): controls risky `exec` commands before execution with `mode` (`ask`/`deny`/`allow`) and `unattended_default` for autonomous sessions. Approval patterns are configurable via `interactive_requires_approval_for`; hard catastrophic patterns remain blocked in `ShellExecTool::check_safety_guards`.

**Hooks** (`[harness.hooks]`): optional **observation** (`[harness.hooks.observation] enabled = true`) appends structured telemetry envelopes to JSONL and/or POSTs to a webhook (bounded queue; optional HMAC signing; `metadata_keys` copies selected inbound metadata for multi-tenant routing). Optional **steering** (`[harness.hooks.steering] enabled = true`) runs subprocess hooks at `user_prompt` (before persisting the user turn), `pre_tool` (after shell policy, before tool execution; can block or replace `args`), and `post_tool` (after each tool; stdout ignored). Implementation: `src/hooks/`, wiring in `src/agent/mod.rs`. Operator reference: `docs/hooks.md`.

**Execution plane** (reproducible code run): **operator guide** → `docs/execution-user-guide.md`; **engineer roadmap** → `docs/execution-implementation-plan.md`. **Keep the user guide in sync** with any change to execution tools, config keys, provider behavior, or defaults—update `docs/execution-user-guide.md` in the **same PR** as the code or config change.

**Phase 0** contracts in `crate::execution`; **Phase 1** `LocalExecutionProvider` in `execution::local`; **Phase 2** gates tools behind **`[harness.execution] enabled = true`** (`ExecutionHarnessConfig` in `config.rs`); **Phase 3** `JupyterExecutionProvider` in `execution::jupyter` when `default_provider = "jupyter"` and `[harness.execution.jupyter]` is set (kernel WebSocket requests **`v1.kernel.websocket.jupyter.org`**, decodes server text + v1 + legacy binary; see user guide); **Phase 4 (MVP)** `SshExecutionProvider` in `execution::ssh` when `default_provider = "ssh"` and `[harness.execution.ssh]` is set (one SSH transport per `execution_session_create`; **Python** uses a persistent framed REPL on a dedicated channel with `mkdir -p`+`cd` before each spawn and a one-retry self-test; **shell** runs `bash -s` per `execution_run` with the same `mkdir -p`+`cd`; `cwd_mode` / `cwd_relative` apply on the **remote** only—see user guide; password via **`SSH_PASSWORD`** or key via `identity_file`; host keys: see user guide); **Phase 5 (MVP)** Colab Session Operator skill (starts Colab VMs via `colab-cli`, supports CPU/GPU/TPU provisioning and script execution). **Phase 6 (artifacts / manifest):** Jupyter materializes `display_data` for PNG/JPEG and large CSV/JSON under **`sandbox_dir/.execution_artifacts/{session_id}/{run_uuid}/`**; `RunResult.attachments` lists sandbox-relative paths. **`execution_artifact_list`** enumerates artifacts per session. Each **`execution_run`** appends one metadata line to **`workspace_dir/.system_generated/execution_runs.jsonl`** (**`run_id`** locates `execution_history/` journals; **`chat_id`** filters the Ratatui **executions** pane per terminal chat thread) and emits **`TelemetryEvent::ExecutionRunFinished`**. Optional **`[harness.execution]`** keys: `artifact_max_file_bytes`, `artifact_max_total_bytes_per_run`, `artifact_max_files_per_run`. **Deferred:** policy-gated execution provisioners (Runpod-style allocation feeding providers) are design-only in **`docs/execution-implementation-plan.md`** (“Deferred — Execution provisioners”). Tools: `execution_session_create`, `execution_run`, `execution_artifact_list`, `execution_cancel`, `execution_session_close`, `execution_env_info` (`src/tools/execution.rs`), wired in `src/main.rs`. Optional keys: `default_provider` (`local`, `jupyter`, `ssh`), `max_output_bytes`, `max_wall_secs`, `max_sessions`, `allowed_providers`, `python_executable` (local host probe / `execution_env_info`; not the remote interpreter for SSH), `local_python_mode` (`repl` default vs `subprocess` / `stateless` / `false` for one interpreter per `execution_run` on **local** Python only), `local_python_runtime` (`system` or `uv_managed`) and UV tuning (`uv_binary`, `uv_python`, `uv_requirements`). Sub-agents with `allowed_tools` must include these names if sub-agents should call them. There is no separate mpsc execution actor yet—the harness holds `Arc<dyn ExecutionProvider>` and tools call it directly.

**Loop hardening:** top-level **`doom_loop_enabled`** in `config.toml` (default true) enables ml-intern-style detection of repeated identical tool calls before each LLM step; a corrective **user** message is persisted when triggered (`src/agent/doom_loop.rs`). Independently, the pure controller in `src/agent/budget.rs` tracks LLM turns, wall time, provider-reported tokens, provider retries, context recoveries, repeated typed tool root causes, and observable progress. It emits run-scoped typed warnings, stops three consecutive failures with the same `(tool, error code)` as `Stuck::RepeatedRootCause`, stops sustained no-progress runs as `Stuck::NoProgress`, and retains `max_iterations` only as the absolute LLM-turn ceiling. Steering and context recovery reset no-progress only; successful new evidence may also clear the repeated-root-cause streak, while consumed time/tokens/retries are never refunded.

**Run-scoped LLM providers:** `AgentLogic` keeps the active provider and its credentials behind one lock and snapshots that pair, plus the filtered failover candidates, when a run is admitted. Active runs and already-admitted FIFO items never read process-global provider state. A `/model` switch atomically changes the pair for subsequent admissions; an inbound accepted after the switch keeps the new pair even when it waits behind an older run. Sub-agents use the same snapshot contract. Embedders that need failover candidates use `AgentLogic::new_with_fallback_providers`; `AgentLogic::new` remains the compatibility path with no candidates. Runtime changes must use `switch_provider_with_credentials`; provider-only and credential-only mutation APIs are intentionally unsupported.

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

2. **Register Tool**: In `src/main.rs`, inject the new initialized instance into the global `ToolRegistry` mapped to `AgentLogic`. 

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
2. Run `cargo clippy --release -p isanagent --all-targets` (or `cargo clippy --all-targets` in debug where supported) and keep Clippy clean. Do not use `#[allow(clippy::…)]` or `allow()` to suppress warnings.
3. Run `cargo fmt` so that it's always well-formatted, avoiding unnecessary diffs that simply come from formatting.
4. This is a living document --keep this document up-to-date as you introduce new features and/or architectures.

## Releases & ALTAI sync
Pushing to `main` runs `.github/workflows/release-artifacts.yml`, which publishes rolling tag **`main-latest`** binaries. After that release job, an optional **`notify-altai-app`** step runs only when repo variable `ALTAI_APP_DISPATCH_ENABLED=true`, and dispatches `isanagent-released` to `altaidevorg/altai-app` when secret `ALTAI_APP_DISPATCH_TOKEN` is set (cross-repo PAT). ALTAI also refreshes the crate from `branch = "main"` on every local/CI/release build via `cargo update -p isanagent`.

## Note for Windows
On windows, building the project in debug mode causes a PDB-related linker error. Build the project in release mode on Windows instead.
