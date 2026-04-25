# Subagent ↔ execution plane wiring (design)

This document plans how **sub-agent tasks** correlate with **execution harness** background jobs. The SQLite table `subagent_tasks` already includes an `execution_job_id` column (`src/memory.rs`); finalization today passes `None` for that field (`src/agent/subagent.rs`).

## Goals

- Let operators answer: “Which `execution_run_background` job did this research subagent start?”
- Keep the **parent** reasoning loop unchanged; correlation is audit metadata, not a second scheduling system.
- Avoid widening every tool trait signature; prefer **task-local** or **session-local** slots already scoped to the subagent’s `ToolExecCtx`.

## Current data flow

1. **Spawn:** `SubagentHarness::spawn` inserts `subagent_tasks` with `task_id`, `parent_chat_id`, `child_chat_id`, prompt, etc.
2. **Run:** `run_reasoning_loop` runs with `ToolExecCtx` keyed by `(channel, child_chat_id, thread_id)` and `is_subagent: true`.
3. **Complete:** `persist_subagent_end(..., execution_job_id: None)` updates the row to terminal status.

Execution tools resolve session from `ToolExecCtx::chat_id` (the child chat). Job IDs are returned in tool JSON when starting background work.

## Recommended wiring (phased)

### Phase A — “Last job” stamp (simplest)

- Add an **`Arc<Mutex<Option<String>>>`** (or `tokio::sync::Mutex`) to `SubagentSpawnDeps`, cloned into a field on `ReasoningLoopCtx` **only for subagents** (or always, unused on parent).
- When `execution_run_background` (or equivalent) **successfully** returns a job id, the tool implementation **locks** that arc and sets `Some(job_id)` (overwrite or append with comma + cap length—product choice).
- When the subagent loop finishes, read the arc, pass `execution_job_id` into `persist_subagent_end`.

**Pros:** Small diff, one obvious job for audit. **Cons:** Multiple concurrent background jobs collapse to one field unless you append with a cap.

### Phase B — Multi-job list

- Same arc, but store `Vec<String>` or JSON array string, dedupe, max 8 ids, truncate for SQLite.
- Optional: merge ids from tool results by parsing assistant-visible tool output (fragile—prefer instrumenting tools).

### Phase C — Telemetry join

- Emit `execution_job_id` on `TelemetryEvent::SubagentFinished` (extend variant) so UIs can filter without opening SQLite.
- Optional inbound metadata on spawn: `parent_execution_job_id` to link a subagent spawned **from** a long-running parent job (orthogonal to child-started jobs).

## Alternatives considered

- **`ToolExecCtx` field:** Works if every tool call receives an updated ctx or a handle; today ctx is cloned per scope—would need `Arc<SubagentExecSidecar>` inside `ToolExecCtx` for interior mutability.
- **Global `DashMap<child_chat_id, JobIds>`:** Easy but harder to test and easy to leak keys; prefer explicit ownership on the spawn future.

## Acceptance checks (when implemented)

- After a subagent calls `execution_run_background` once, `task_history_list` / SQLite row shows non-null `execution_job_id` when the job id was returned.
- Parent-only runs leave `execution_job_id` null.
- No change to ordering of assistant/tool messages in memory.
