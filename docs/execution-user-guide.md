# Code execution harness — user guide

This guide is for **operators and users** of isanagent who want to run code safely inside the agent workspace. For the internal roadmap and trait design, see **`execution-implementation-plan.md`**.

## What you get

When **`[harness.execution] enabled = true`**, the agent gains five tools:

| Tool | Purpose |
|------|--------|
| **`execution_session_create`** | Start a sandbox-scoped session (choose language: Python, shell, etc.). |
| **`execution_run`** | Run code in that session (timeouts and output size are capped). |
| **`execution_cancel`** | Best-effort interrupt of a long-running execution (when the provider supports it). |
| **`execution_session_close`** | Tear down the session and release resources. |
| **`execution_env_info`** | Show provider capabilities and (for local Python) try `python -V`. |

Two providers are implemented today:

- **`local`** — each session uses a working directory under your workspace sandbox and runs subprocesses (e.g. `python -u -c …`).
- **`jupyter`** — each session is a **Jupyter Server** kernel you point at with `base_url` + token; runs use the kernel’s WebSocket execute channel (persistent variables, interrupt via server API).

**SSH** and other remotes are planned separately.

## Enable the feature

In your workspace **`config.toml`** (next to `.agents/`, not inside the sandbox):

```toml
[harness.execution]
enabled = true
```

Optional keys (defaults are sensible if omitted):

| Key | Meaning |
|-----|--------|
| `default_provider` | **`local`** (subprocess) or **`jupyter`** (remote kernel). |
| `max_wall_secs` | Upper bound on each run’s `timeout_secs` (default 300, max 86400). |
| `max_output_bytes` | Max combined stdout+stderr per run (default 256 KiB). |
| `max_sessions` | Max concurrent sessions (default 32). |
| `allowed_providers` | e.g. `["local"]` or `["jupyter"]`; if empty or omitted, any implemented provider is allowed. |
| `python_executable` | Command for **local** Python runs and `execution_env_info` (default `python`). Ignored for Jupyter execution. |

When `default_provider = "jupyter"`, add **`[harness.execution.jupyter]`**:

| Key | Meaning |
|-----|--------|
| `base_url` | Jupyter Server root, e.g. `http://127.0.0.1:8888` (no `/lab` path). **Required** for Jupyter. |
| `token` | Optional server token. Prefer host env **`JUPYTER_TOKEN`** (wins over this field) so secrets are not committed. |
| `kernel_name` | Kernel spec name for `POST /api/kernels` when `language` is Python or unset (default **`python3`**). |

Restart the agent after editing config.

## Workspace layout (important)

- **`workspace_dir`** (outer): holds `config.toml`, logs, `.system_generated/`, etc.
- **`sandbox_dir`** (inner): usually `workspace_dir/workspace` — this is where execution runs and where paths are resolved.

Filesystem tools and execution share the same **sandbox boundary** when `restrict_to_workspace = true` (default). Do not put secrets in the sandbox if the model can read them.

## Typical workflow (for you or the model)

1. **`execution_session_create`** — optional `label`, optional `language`.  
   - **`local`:** `python`, `py`, `shell`, `sh`, `bash`.  
   - **`jupyter`:** `python` / `py` / unset (uses `kernel_name`), or **`r`** / **`R`** (uses the **`ir`** kernel spec if installed).  
   - Response includes **`session_id`** and capability summaries — keep the `session_id` for the next steps.

2. **`execution_run`** — required: `session_id`, `code`. Optional: `timeout_secs`, `cwd_mode` (`session_default` or `sandbox_relative`), and `cwd_relative` when using `sandbox_relative`.  
   - **`jupyter`:** only **`session_default`** is supported for `cwd_mode` (no per-run sandbox cwd); use notebook magics such as `%cd` inside `code` if you must change directory on the server.

3. When finished (or to free slots): **`execution_session_close`** with the same `session_id`.

Use **`execution_cancel`** if a run is stuck and the provider reports **`supports_interrupt`** (true for **`local`** and **`jupyter`**).

## Python and virtual environments (local provider)

The **local** harness runs **`python_executable`** as a normal process. It does **not** auto-activate a venv; you should either:

- Set **`python_executable`** to the interpreter you want (e.g. path to `uv`-managed `.venv\Scripts\python.exe` on Windows, or `.../.venv/bin/python` on Unix), or  
- Rely on a shell session (`language: shell`) and invoke `uv run …` / activate scripts in **`code`** (understand the security tradeoff of shell mode).

If something fails, check **`execution_env_info`** and the tool error text (missing interpreter, timeout, etc.).

For **Jupyter**, pick the kernel environment by **`kernel_name`** and the kernels installed on that server; the agent does not configure `pip` or conda from tool args in this release.

**Notebook vs Lab:** both use **Jupyter Server**; the kernel WebSocket URL and message framing are the same. Use **`base_url`** as the server root (for example `http://127.0.0.1:8888`), not the `/lab?token=…` UI URL—put the token in **`JUPYTER_TOKEN`** or `[harness.execution.jupyter].token` instead.

**Output capture:** the server may send `print()` output as **JSON text** WebSocket frames or as **binary v1** frames (depending on subprotocol). The agent collects **`stream`** (stdout/stderr), **`execute_result`** / **`display_data`** (`text/plain`), **`error`**, and **`execute_reply`**. Bare expressions (last line without `print`) appear via **`execute_result`**, not always as a `stream`.

## Working directory for a run

- **`cwd_mode`: `session_default`** (default) — run in the session’s root (sandbox root).  
- **`cwd_mode`: `sandbox_relative`** — requires **`cwd_relative`** (e.g. `pkg`); resolved under the sandbox like other tools.

## Sub-agents

If you use **`[harness.subagents]`** with **`allowed_tools`**, include the execution tool names explicitly if sub-agents should run code:

`execution_session_create`, `execution_run`, `execution_cancel`, `execution_session_close`, `execution_env_info`

## Limits and safety

- Runs are **time-bounded** and **output-bounded**; huge prints are truncated with a marker in the output.  
- **`execution_cancel`** uses process kill / `taskkill` best effort on Windows.  
- Treat **`shell`** mode like **`exec`**: only enable paths and prompts you trust.

## Terminal UI

Start the binary with your workspace, for example:

```bash
cargo run --release -p isanagent -- --workspace /path/to/my_agent
```

Ensure your **`[provider]`** API key env is set. The model should see the execution tools in its tool list once enabled.

## Roadmap (where this doc stays in sync)

- **Implemented:** Jupyter provider (`execution-implementation-plan.md` Phase 3) — same tool names, `default_provider = "jupyter"`, `[harness.execution.jupyter]`.  
- **Later:** SSH and other remotes in separate PRs.

When we add providers or config keys, this guide and **`AGENTS.md`** should be updated in the same change so operators are not surprised.
