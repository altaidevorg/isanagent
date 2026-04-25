---
name: execution-research
description: Run numeric or ML code through isanagent execution tools without flooding context—artifacts, incremental runs, sandbox paths.
requires:
  bins: []
  env: []
always: false
---

# Execution research (isanagent)

Use this playbook whenever you are writing or running analysis code, plots, tables, or benchmarks inside the workspace execution harness.

## Before you run anything

1. Call **`execution_env_info`** once. Note `default_provider`, `max_output_bytes`, **`max_wall_secs`**, **`default_run_timeout_secs`**, **`timeout_policy`**, and artifact caps. If execution is disabled in config.toml (`[harness.execution] enabled = false`), stop and tell the user to turn it back on—do not invent a workaround.
2. Plan for **stdout/stderr caps**: each stream gets roughly half of `max_output_bytes` (with a floor). Large prints truncate; design outputs accordingly.

## Session lifecycle (strict order)

1. **`execution_session_create`** — set `language` (`python`, `shell`, or omit for default Python). Keep **one active run or background job per session**; wait for each to finish before starting another on the same `session_id`.
2. **`execution_run`** — pass `session_id`, source `code`, and `timeout_secs` within `max_wall_secs`. Add **`description`** for any non-trivial run so the terminal strip and JSONL audits show intent. Prefer **small, verifiable steps** (import → load → transform → plot) instead of one giant cell. For **long** wall clocks (many minutes or hours), prefer **`execution_run_background`** (with **`description`**) and poll **`execution_job_status`** / **`execution_job_result`** (see skill **`long-running-execution`**) so the agent thread is not blocked for the entire duration.
3. **`execution_artifact_list`** — when you need filenames or sizes under **`.execution_artifacts/<session_id>/`** (Jupyter materialized payloads, or files your code wrote there on local runs).
4. **`glob_files`** / **`search_text`** — inspect text logs, CSV snippets, or small previews **by path**; never pull multi‑MB content into the reply.
5. **`execution_session_close`** with the same `session_id` when the analysis branch is done so slots are freed.

## Where outputs land

- **Jupyter provider:** binary or large `display_data` / similar payloads become files under **`sandbox_dir/.execution_artifacts/<session_id>/<run_uuid>/`**. The tool response includes **`attachments`** (sandbox-relative paths, MIME). Treat those paths as the source of truth; summarize, do not inline blobs.
- **Local Python:** by default the interpreter is **persistent per session** (variables survive across `execution_run` until timeout, cancel, cwd change, or close). If you need a clean interpreter each time, that is a **host config** choice (`local_python_mode = subprocess` in `config.toml`)—you cannot toggle it from a tool. For plots/tables, still **`savefig`** / write files under **`.execution_artifacts/...`** or another sandbox path you resolve with normal file tools so results stay on disk.

## Anti-patterns (avoid)

- Dumping base64 or huge JSON into chat.
- Issuing **identical** `execution_run` payloads in a tight loop when results do not change—isanagent may inject a **doom-loop** corrective user message; change code, parameters, or diagnostics instead.
- Skipping **`execution_session_close`** after a long exploratory branch.

## If something hangs

Use **`execution_cancel`** on that `session_id`, then decide whether to retry with a shorter timeout, smaller data, or different approach.
