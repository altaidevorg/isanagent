# Code execution harness — user guide

This guide is for **operators and users** of isanagent who want to run code safely inside the agent workspace. For the internal roadmap and trait design, see **`execution-implementation-plan.md`**.

## What you get

When the execution harness is **not** turned off in config (it is **on by default**), the agent gains these tools:

| Tool | Purpose |
|------|--------|
| **`execution_session_create`** | Start a sandbox-scoped session (choose language: Python, shell, etc.). |
| **`execution_run`** | Run code in that session **synchronously** (timeouts and output size are capped). Optional **`description`** (short human summary) improves the terminal execution strip and audit JSONL. Returns **`attachments`** when Jupyter (or future providers) materialize binary or large text blobs on disk. |
| **`execution_run_background`** | Same as **`execution_run`** plus returns a **`job_id`** immediately. Optional **`label`** (logs) and **`description`** (UI/audits). Use for long ML jobs so the model is not blocked for the full wall clock. |
| **`execution_job_status`** | Poll a background job: status, timestamps, error text. |
| **`execution_job_result`** | When the job is finished, fetch **`RunResult`** JSON (truncated to the session **`max_tool_output_chars`** cap). |
| **`execution_job_list`** | List in-memory background jobs (optional **`session_id`** filter). |
| **`execution_job_cancel`** | Best-effort interrupt by **`job_id`** (same capability rules as **`execution_cancel`**). |
| **`execution_artifact_list`** | List files under `.execution_artifacts/<session_id>/` for that session (paths relative to sandbox). |
| **`execution_cancel`** | Best-effort interrupt of the current run for a **`session_id`** (when the provider supports it). |
| **`execution_session_close`** | Tear down the session and release resources. |
| **`execution_env_info`** | Show provider capabilities, artifact caps, **`max_wall_secs`**, **`default_run_timeout_secs`**, a **`timeout_policy`** reminder, and (for local Python) try `python -V`. |

### Research helpers (related tools)

The agent also ships read-only **arXiv** (`arxiv_search`, `arxiv_fetch`) and **Hugging Face Hub file** (`hf_hub_file_fetch`, uses host env **`HF_TOKEN`** when set) tools. Use them together with **`web_fetch`** on stable URLs—e.g. `https://raw.githubusercontent.com/.../refs/heads/main/...` for pinned examples—when checking current library APIs before long **`execution_run_background`** jobs.

Three providers are implemented today:

- **`local`** — each session uses a working directory under your workspace sandbox. **Python (default):** one long-lived interpreter per session (**REPL-like**): variables and imports persist across **`execution_run`** calls until the session closes, you cancel, a run times out, or the working directory for the run changes (then the interpreter is restarted in the new cwd). Code is sent over a framed stdin/stdout channel to the worker (not via argv). **Opt out** with **`local_python_mode = "subprocess"`** in **`[harness.execution]`** to use a fresh **`python -u -`** subprocess per run (legacy, stateless). **Runtime choice:** `local_python_runtime = "uv_managed"` (default) provisions and reuses a managed env under `workspace_dir/.system_generated/uv/envs/`; `local_python_runtime = "system"` requires explicitly setting `python_executable`. If uv is missing at startup and uv-managed runtime is active, terminal mode prompts for yes/no auto-install and `/install-python` can be used later. **Shell** sessions still use one short-lived **`sh -c`** / **`cmd /C`** per run. Stdout/stderr are capped the same as **`max_output_bytes`** (half per stream, minimum each side). On Unix the child is placed in its own process group and cancellation/timeout sends **SIGKILL** to that group (similar to Windows **`taskkill /T`**); **`SIGKILL`** is never sent for PID 0 or 1.
- **`jupyter`** — each session is a **Jupyter Server** kernel you point at with `base_url` + token; runs use the kernel’s WebSocket execute channel (persistent variables, interrupt via server API). **`display_data` / `execute_result`** may include **`image/png`**, **`image/jpeg`**, large **`text/csv`**, or large **`application/json`** payloads: those are written under **`sandbox_dir/.execution_artifacts/<session_id>/<run_uuid>/`** (size-capped) and referenced in **`RunResult.attachments`**; stdout gets short `[execution artifact] …` lines. Use **`execution_artifact_list`** to browse.
- **`ssh`** — **`execution_session_create`** opens one authenticated SSH session (TCP + handshake) to the configured host and keeps it open for that session; each **`execution_run`** opens a new exec channel, runs a short remote `cd … && exec python3 -u -` (or `exec bash -s` for shell), and **streams your code on channel stdin** so large payloads are not embedded in `argv`. There is **no** Jupyter-style persistent kernel variables across runs—only the transport is reused. **`execution_cancel`** only cancels the client wait (remote process may keep running). Use **`identity_file`** (OpenSSH private key) and/or host env **`SSH_PASSWORD`** (never commit passwords in `config.toml`).

For **Google Colab**, use the **`colab-cli`** skill (invoke `colab` commands via `exec`) instead of the built-in execution harness providers. This allows more flexible management of Colab VMs including GPU/TPU provisioning.

## Configuration

Execution is **on by default** with **`default_provider = "local"`** when keys are omitted. To disable the harness, set **`[harness.execution] enabled = false`** in workspace **`config.toml`** (next to `.agents/`, not inside the sandbox). If you restrict **`allowed_providers`**, set **`default_provider`** to a member of that list (for example **`jupyter`**).

Optional keys (defaults are sensible if omitted):

| Key | Meaning |
|-----|--------|
| `default_provider` | **`local`** (default), **`jupyter`** (remote kernel), or **`ssh`** (remote exec over SSH). |
| `max_wall_secs` | Upper bound on each run’s **`timeout_secs`** (default **3600**, clamped **1–86400** seconds = up to 24h). Raise this when you need longer blocking or background runs. |
| `default_execution_timeout_secs` | Default wall clock when the model omits **`timeout_secs`** on **`execution_run`** / **`execution_run_background`** (default **600**, clamped to **`max_wall_secs`**). |
| `max_output_bytes` | Max combined stdout+stderr per run (default 256 KiB). |
| `max_sessions` | Max concurrent sessions (default 32). |
| `allowed_providers` | e.g. `["local"]`, `["jupyter"]`, `["ssh"]`; if empty or omitted, any implemented provider is allowed. |
| `python_executable` | Required only when `local_python_runtime = "system"` (explicit host interpreter path/command). Ignored for UV-managed local runtime. For **SSH**, the remote interpreter is **`[harness.execution.ssh].remote_python`** (default `python3`). |
| `local_python_mode` | **`repl`** (default, or any value other than the opt-outs below): one **local** Python interpreter per session. **`subprocess`**, **`fresh`**, **`stateless`**, **`one_shot`**, **`false`**, **`0`**: each **`execution_run`** starts a new **`python -u -`** process (no shared namespace). |
| `local_python_runtime` | **`uv_managed`** (default) provisions/caches a runtime with `uv` under `.system_generated/uv/envs/`; **`system`** requires explicit `python_executable`. |
| `uv_binary` | Command used for UV-managed env creation/install (default `uv`). |
| `uv_python` | Python version string for `uv venv --python` when UV-managed runtime is enabled (default `3.11`). |
| `uv_requirements` | Optional package specs installed once into UV-managed runtime (example: `["numpy", "pandas>=2.2"]`). |
| `artifact_max_file_bytes` | Max bytes per saved artifact file (default 4 MiB, clamped 64 KiB–64 MiB). |
| `artifact_max_total_bytes_per_run` | Max total bytes for all artifacts in one `execution_run` (default 32 MiB). |
| `artifact_max_files_per_run` | Max artifact files per run (default 64, clamped 1–256). |
| `wake_on_job_terminal` | When **true** (default), a background job reaching a **terminal** state (completed, failed, timeout, or cancelled) enqueues a **synthetic inbound** message to the same chat so the model can call **`execution_job_status`** / **`execution_job_result`** without waiting for the user. Set **`false`** for API-only or headless integrations that must not auto-start another reasoning turn. Inbound metadata includes **`isanagent_synthetic_job_followup`** and **`execution_job_id`**. |

Top-level **`doom_loop_enabled`** (optional, default **true**): when true, the agent detects repeated identical tool calls and injects a corrective user message before the next LLM call (see `src/agent/doom_loop.rs`).

Each successful **`execution_run`** or completed **`execution_run_background`** job also appends one JSON line to **`workspace_dir/.system_generated/execution_runs.jsonl`** (metadata only: no code body; may include **`job_id`** and optional **`description`**; background lines may include **`job_id`**) and emits **`ExecutionRunFinished`** telemetry (optional **`description`**). When a background job reaches a **finished** state (completed, failed, cancelled, or timeout), the agent also emits **`ExecutionJobFinished`** telemetry and appends **`workspace_dir/.system_generated/execution_jobs.jsonl`** (metadata audit; optional **`description`**). A user-visible **outbound** notice is sent, and (unless **`wake_on_job_terminal = false`**) a synthetic **inbound** is enqueued to continue the turn—same mechanism as **`cron`**-scheduled reminders.

Additionally, every **`execution_run`** (all providers) writes a **run journal** under **`workspace_dir/.system_generated/execution_history/{provider}/{session_id}/{run_id}/`**: **`run.json`** (truncated stdout/stderr, attachment list, timestamps) and **`source.txt`** (the exact code run). Treat journals as potentially sensitive if code contained secrets.

When `default_provider = "jupyter"`, add **`[harness.execution.jupyter]`**:

| Key | Meaning |
|-----|--------|
| `base_url` | Jupyter Server root, e.g. `http://127.0.0.1:8888` (no `/lab` path). **Required** for Jupyter. |
| `token` | Optional server token. Prefer host env **`JUPYTER_TOKEN`** (wins over this field) so secrets are not committed. |
| `kernel_name` | Kernel spec name for `POST /api/kernels` when `language` is Python or unset (default **`python3`**). |
| `notebook_sync_path_template` | Optional. When set (e.g. `isanagent/{session_id}.ipynb`), each successful run **appends a code cell** to that server-side notebook via the Contents API (`{session_id}` is the **sanitized** isanagent session id). Open it in JupyterLab to watch progress. |

When `default_provider = "ssh"`, add **`[harness.execution.ssh]`**:

| Key | Meaning |
|-----|--------|
| `host` | Remote hostname or IP. **Required** for SSH. |
| `port` | SSH port (default **22**). |
| `user` | Remote login name. **Required** for SSH. |
| `identity_file` | Path to an OpenSSH **private** key (optional if **`SSH_PASSWORD`** is set in the agent process environment). Tilde (`~`) expansion is applied. |
| `remote_workdir` | **Absolute** path on the remote host (POSIX, e.g. `/home/you/isanagent-runs`). Only letters, digits, `/`, `_`, `-`, `.`; no `..`. **Required**. |
| `remote_python` | Remote Python executable for `language: python` (default **`python3`**). |
| `accept_unknown_host_keys` | Default **true**: accept any server host key (**vulnerable to MITM** on untrusted networks). Set **false** to fail closed until strict host-key verification exists. |

## Workspace layout (important)

- **`workspace_dir`** (outer): holds `config.toml`, logs, `.system_generated/`, etc.
- **`sandbox_dir`** (inner): usually `workspace_dir/workspace` — this is where execution runs and where paths are resolved.

Filesystem tools and execution share the same **sandbox boundary** when `restrict_to_workspace = true` (default). Do not put secrets in the sandbox if the model can read them.

Materialized run artifacts live under **`sandbox_dir/.execution_artifacts/`** (session segment is sanitized for path safety). Operator scenarios: **`docs/execution-use-cases.md`**.

## Typical workflow (for you or the model)

1. **`execution_session_create`** — optional `label`, optional `language`, optional **`resume_jupyter_kernel_id`** (Jupyter only).  
   - **`local`:** `python`, `py`, `shell`, `sh`, `bash`.  
   - **`jupyter`:** `python` / `py` / unset (uses `kernel_name`), or **`r`** / **`R`** (uses the **`ir`** kernel spec if installed).  
   - **`ssh`:** `python` / `py` / unset, or `shell` / `sh` / `bash`.  
   - Response includes **`session_id`** and capability summaries — keep the `session_id` for the next steps. Jupyter responses include **`jupyter_kernel_id`** (and **`jupyter_notebook_sync_path`** when `notebook_sync_path_template` is configured).

2. **`execution_run`** — required: `session_id`, `code`. Optional: `timeout_secs`, **`description`** (short human summary for the terminal strip and `execution_runs.jsonl`), `cwd_mode` (`session_default` or `sandbox_relative`), and `cwd_relative` when using `sandbox_relative`. Call **`execution_env_info`** first in a session if you need exact **`max_wall_secs`** / **`default_run_timeout_secs`**.  
   - **`jupyter`:** only **`session_default`** is supported for `cwd_mode` (no per-run sandbox cwd); use notebook magics such as `%cd` inside `code` if you must change directory on the server.  
   - **`ssh`:** `cwd_mode` applies on the **remote host** (not the agent workspace). **`session_default`** uses **`[harness.execution.ssh].remote_workdir`**. **`sandbox_relative`** means: if `cwd_relative` starts with `/`, use that absolute path on the remote (same path character rules as `remote_workdir`); otherwise treat `cwd_relative` as a path under `remote_workdir` (no `..` segments). Before every run the provider runs **`mkdir -p`** for that remote cwd, then **`cd`**, so a missing `remote_workdir` no longer fails shell or Python startup. Python sessions use a **persistent REPL** (same framed worker as local): variables and imports survive across `execution_run` calls until you change remote cwd or the run errors/times out; the REPL performs a short self-test when opened and retries once on failure. Shell mode still runs a fresh `bash -s` per run. For unattended connects to new hosts, set **`accept_unknown_host_keys = true`** (understand the MITM tradeoff) or pre-populate known_hosts—otherwise the TCP session may hang waiting for a host-key prompt that never reaches the agent.

3. **Long runs (optional):** use **`execution_run_background`** with the same arguments (plus optional **`label`** and recommended **`description`**). Poll **`execution_job_status`** until **`terminal`** is true, then read **`execution_job_result`**. Jobs are **process-local** (lost if the agent exits). Only **one** active run or background job may use a session at a time; for overlapping long work, use **separate execution sessions** (or providers that allow it).

4. When finished (or to free slots): **`execution_session_close`** with the same `session_id`.

Use **`execution_cancel`** (by session) or **`execution_job_cancel`** (by `job_id`) if a run is stuck and the provider reports **`supports_interrupt`** (true for **`local`** and **`jupyter`**; **false** for **`ssh`** in the current release).

## Python and virtual environments (local provider)

The **local** harness runs Python in one of two runtime modes:

- **`local_python_runtime = "uv_managed"`** (default): provisions/reuses a managed interpreter under `.system_generated/uv/envs/` using `uv venv`, then executes runs with that interpreter.
- **`local_python_runtime = "system"`**: runs **`python_executable`** as a normal process (no automatic venv activation). `python_executable` must be set explicitly in config.

The first time a UV-managed environment is created (and when `uv_requirements` triggers `uv pip install`), work can take tens of seconds. During that time the **terminal** tool strip shows short status lines, and **`POST /v1/responses`** with **`stream: true`** emits **`tool_progress`** SSE events so the UI does not look stuck.

For `system` runtime, you should either:

- Set **`python_executable`** to the interpreter you want (e.g. path to `uv`-managed `.venv\Scripts\python.exe` on Windows, or `.../.venv/bin/python` on Unix), or  
- Rely on a shell session (`language: shell`) and invoke `uv run …` / activate scripts in **`code`** (understand the security tradeoff of shell mode).

If uv-managed runtime is enabled and `uv` is not found on `PATH` at launch, terminal mode asks whether to auto-install uv (`yes`/`no`). You can also run `/install-python` in the terminal UI at any time.

If something fails, check **`execution_env_info`** and the tool error text (missing interpreter, timeout, etc.).

For **Jupyter**, pick the kernel environment by **`kernel_name`** and the kernels installed on that server; the agent does not configure `pip` or conda from tool args in this release.

## Sub-agents

If you use **`[harness.subagents]`** with **`allowed_tools`**, include the execution tool names explicitly if sub-agents should run code:

`execution_session_create`, `execution_run`, `execution_run_background`, `execution_job_status`, `execution_job_result`, `execution_job_list`, `execution_job_cancel`, `execution_artifact_list`, `execution_cancel`, `execution_session_close`, `execution_env_info`

## Limits and safety

- Runs are **time-bounded** and **output-bounded**; huge prints are truncated with a marker in the output.  
- **`execution_cancel`** / **`execution_job_cancel`** use process kill / `taskkill` best effort on Windows.  
- Background jobs are retained in memory for polling until evicted when the in-process registry is full (completed jobs are dropped oldest-first).  
- Treat **`shell`** mode like **`exec`**: only enable paths and prompts you trust.

## Roadmap (where this doc stays in sync)

- **Implemented:** Jupyter provider (`execution-implementation-plan.md` Phase 3); SSH MVP (`execution-implementation-plan.md` Phase 4); UV-managed local runtime; Phase 6 artifacts, **`execution_artifact_list`**, run manifest (`execution_runs.jsonl`), telemetry **`ExecutionRunFinished`**, background jobs (**`execution_run_background`**, **`execution_jobs.jsonl`**, **`ExecutionJobFinished`**), and **`doom_loop_enabled`**.  
- **Later:** OAuth-native Colab integration feasibility output and execution provisioners (deferred design doc).

When we add providers or config keys, this guide and **`AGENTS.md`** should be updated in the same change so operators are not surprised.
