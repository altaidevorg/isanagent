# Handover — harness work (Phases 1–5) + SQLite todos

## Summary

This branch delivers harness **Phases 1–5** (through sub-agents / plans under `[harness.subagents]`). **Phase 6** in `docs/harness-implementation-plan.md` is **not** implemented.

## What Was Done

- **Phase 1:** `glob_files`, `search_text` (rg or fallback), `edit_file` (`replace_all`, diff snippet); glob root/canonicalize fix for Windows `\\?\` paths (`src/tools/builtin.rs`).
- **Phase 2:** `todo_write` → table `harness_todos` in `agent_memory.db`; `search_tools` + registration-order catalog; `load_skill_instructions` list/metadata (`src/tools/workflow.rs`, `src/tools.rs`, `src/agent/mod.rs`, `src/skills.rs`, `src/memory.rs`, `src/main.rs`).
- **Concurrency:** `configure_agent_sqlite_connection` + `AGENT_SQLITE_BUSY_TIMEOUT_MS` on memory + todo connections (`src/memory.rs`, `src/tools/workflow.rs`).
- **Phase 3:** `ClarificationHub` (`src/clarification.rs`), `ToolExecCtx` task-local (`src/tool_runtime.rs`), `AskUserTool` (`src/tools/workflow.rs`), `AgentLogic` inbound **early** `try_deliver_reply` before cancel/spawn, `ToolCallRuntime` + scoped tool execution (`src/agent/mod.rs`), terminal `[Question]` for `isanagent_clarification` metadata (`src/channels/terminal.rs`), registration in `src/main.rs`.
- **Phase 4:** `GitWorktreeTool` (`src/tools/builtin.rs`), `[harness.git_worktree]` in `src/config.rs`, conditional registration in `src/main.rs`.
- **Phase 5:** `SubagentHarness` + tools in `src/agent/subagent.rs`; `ReasoningLoopCtx` / scoped tool execution; `[harness.subagents]` (`cancel_children_on_parent_cancel`, `allowed_tools`, `max_tasks`, `max_wait_secs`); agent split `src/agent/mod.rs`.
- **ML engineer convergence:** `[harness.ml_engineer]` + `assets/ml_engineer_overlay.md` (`src/ml_engineer.rs`); optional `workspace/ML_POLICY.md` in `compile_system_prompt` (`src/workspace.rs`); onboard writes `workspace/ML_ENGINEER_OVERLAY.md` as a human-readable copy; harness/runtime lines in `run_reasoning_loop`; optional forbid-final-without-tools (config + inbound metadata); `subagent_tasks` SQLite + `task_history_list`; `arxiv_*` / `hf_hub_file_fetch` in `src/tools/ml_domain.rs`; parallel safe tool batch in main loop; onboarding skills `ml-execution-preflight`, `literature-to-recipe`, `oom-recovery-playbook`. Subagent ↔ execution job correlation: `docs/subagent-execution-wiring.md`.
- **Docs:** `docs/harness-implementation-plan.md`, `AGENTS.md` updated for the above.

## What We Tried / What Didn’t Work

- **Tool trait context:** Session identity for `ask_user` is **not** passed through `Tool::execute(args)`; it uses **`tokio::task_local`** instead of extending the trait (avoids touching every tool signature).
- **Empty clarification replies:** No re-prompt loop; `allow_empty: false` returns an error after one empty reply (documented in Phase 3 acceptance).

## Bugs & Fixes

- **Glob on Windows:** `strip_prefix` vs canonical paths — fixed with canonical walk root / path helper in builtin glob tests.
- **Clippy `too_many_arguments`:** Bundled `tool_exec_ctx` + `clarification_hub` into **`ToolCallRuntime`** for `execute_tool_call_with_activity`.
- **rusqlite `busy_timeout`:** Returns `Result` in 0.38 — propagated through `configure_agent_sqlite_connection`.

## Key Decisions (and Why)

- **One DB for todos + memory:** Single file to copy/backup; schema via `ensure_harness_todos_schema` in `memory.rs`.
- **Clarification before cancel on inbound:** Default behavior cancels the prior task by `chat_id`; routing clarification first avoids killing the task that is blocked inside `ask_user`.
- **Session key:** `channel:chat_id:thread` (empty thread segment when `thread_id` is `None`) — aligns with `SessionManager` / memory session key.

## Gotchas / Things to Watch Out For

- **`ask_user` outside agent tool scope:** Fails with a clear error if `ToolExecCtx` is not set (e.g. calling the tool without the agent’s scoped execution).
- **Cancellation is still keyed by `chat_id` only** for `BusMessage::Cancel` / `cancellation_tokens` (pre-existing); clarification matching uses full **session key** including channel and thread — keep them consistent when adding channels.
- **Clippy:** Run `cargo clippy --release -p isanagent --all-targets` before merge; keep the tree warning-free without `allow()` suppressions.
- **Windows:** Prefer `cargo build/test --release` (see `AGENTS.md`).

## Next Steps

- [ ] Merge or park this branch; open the **other PR** as planned.
- [ ] When resuming harness: **Phase 6** (notebook, LSP, MCP, remote) per `docs/harness-implementation-plan.md`; **execution plane** (sessions, providers, capabilities) per `docs/execution-implementation-plan.md`.
- [ ] Optional hardening: API/UI explicitly handle `isanagent_clarification` in SSE or REST responses (currently same as other outbounds + metadata).

## Important Files Map

| Path | Purpose |
|------|---------|
| `docs/harness-implementation-plan.md` | Phase checklist and acceptance criteria |
| `src/config.rs` | `AppConfig`, optional `[harness.git_worktree]` |
| `AGENTS.md` | Architecture, sandbox, todos DB, `ask_user` behavior |
| `src/clarification.rs` | `ClarificationHub`, `METADATA_CLARIFICATION` |
| `src/tool_runtime.rs` | `ToolExecCtx`, task-local scope for tools |
| `src/agent/mod.rs` | Inbound clarification routing; `ToolCallRuntime`; `execute_tool_call_with_activity`; `ReasoningLoopCtx` |
| `src/agent/subagent.rs` | `SubagentHarness`, spawn/plan/task tools |
| `src/tools/workflow.rs` | `todo_write`, `search_tools`, `AskUserTool`, todo SQLite helpers |
| `src/tools/builtin.rs` | Glob/search/edit/message/shell/web/memory/`git_worktree` tools |
| `src/tools.rs` | `ToolRegistry`, catalog, `search_tool_index` |
| `src/memory.rs` | Memory actor, `harness_todos` + `subagent_tasks` schema, SQLite busy_timeout helper |
| `src/ml_engineer.rs` | ML overlay text, subagent research append |
| `docs/subagent-execution-wiring.md` | Plan for `subagent_tasks.execution_job_id` + execution tools |
| `src/tools/ml_domain.rs` | `arxiv_search`, `arxiv_fetch`, `hf_hub_file_fetch` |
| `src/channels/terminal.rs` | `[Question]` styling for clarification outbounds |
| `src/main.rs` | Tool registration, `ClarificationHub` + `AgentLogicParams` wiring |
| `src/execution/` | Phases 0–2: contracts, `local.rs`, `harness.rs` + `build_execution_harness`; tools in `src/tools/execution.rs`; `[harness.execution]` in `config.rs` (`docs/execution-implementation-plan.md`) |

## Run/Test Commands

```powershell
cd C:\Users\Yusuf\agent-rs
cargo fmt
cargo clippy --release -p isanagent --all-targets
cargo test --release -p isanagent
```
