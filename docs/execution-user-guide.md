# Code execution harness — user guide

This guide is for **operators and users** of isanagent who want to run code safely inside the agent workspace. For the internal roadmap and trait design, see **`execution-implementation-plan.md`**.

## What you get

When **`[harness.execution] enabled = true`**, the agent gains six tools:

| Tool | Purpose |
|------|--------|
| **`execution_session_create`** | Start a sandbox-scoped session (choose language: Python, shell, etc.). |
| **`execution_run`** | Run code in that session (timeouts and output size are capped). Returns **`attachments`** when Jupyter (or future providers) materialize binary or large text blobs on disk. |
| **`execution_artifact_list`** | List files under `.execution_artifacts/<session_id>/` for that session (paths relative to sandbox). |
| **`execution_cancel`** | Best-effort interrupt of a long-running execution (when the provider supports it). |
| **`execution_session_close`** | Tear down the session and release resources. |
| **`execution_env_info`** | Show provider capabilities, artifact caps, and (for local Python) try `python -V`. |

Three providers are implemented today:

- **`local`** — each session uses a working directory under your workspace sandbox. **Python (default):** one long-lived interpreter per session (**REPL-like**): variables and imports persist across **`execution_run`** calls until the session closes, you cancel, a run times out, or the working directory for the run changes (then the interpreter is restarted in the new cwd). Code is sent over a framed stdin/stdout channel to the worker (not via argv). **Opt out** with **`local_python_mode = "subprocess"`** in **`[harness.execution]`** to use a fresh **`python -u -`** subprocess per run (legacy, stateless). **Shell** sessions still use one short-lived **`sh -c`** / **`cmd /C`** per run. Stdout/stderr are capped the same as **`max_output_bytes`** (half per stream, minimum each side). On Unix the child is placed in its own process group and cancellation/timeout sends **SIGKILL** to that group (similar to Windows **`taskkill /T`**); **`SIGKILL`** is never sent for PID 0 or 1.
- **`jupyter`** — each session is a **Jupyter Server** kernel you point at with `base_url` + token; runs use the kernel’s WebSocket execute channel (persistent variables, interrupt via server API). **`display_data` / `execute_result`** may include **`image/png`**, **`image/jpeg`**, large **`text/csv`**, or large **`application/json`** payloads: those are written under **`sandbox_dir/.execution_artifacts/<session_id>/<run_uuid>/`** (size-capped) and referenced in **`RunResult.attachments`**; stdout gets short `[execution artifact] …` lines. Use **`execution_artifact_list`** to browse.
- **`ssh`** — **`execution_session_create`** opens one authenticated SSH session (TCP + handshake) to the configured host and keeps it open for that session; each **`execution_run`** opens a new exec channel, runs a short remote `cd … && exec python3 -u -` (or `exec bash -s` for shell), and **streams your code on channel stdin** so large payloads are not embedded in `argv`. There is **no** Jupyter-style persistent kernel variables across runs—only the transport is reused. **`execution_cancel`** only cancels the client wait (remote process may keep running). Use **`identity_file`** (OpenSSH private key) and/or host env **`SSH_PASSWORD`** (never commit passwords in `config.toml`).

Other hosted remotes (Colab-shaped providers, policy-gated **provisioners** that allocate targets) are described in **`execution-implementation-plan.md`** and are not the same as this SSH provider.

## Enable the feature

In your workspace **`config.toml`** (next to `.agents/`, not inside the sandbox):

```toml
[harness.execution]
enabled = true
```

Optional keys (defaults are sensible if omitted):

| Key | Meaning |
|-----|--------|
| `default_provider` | **`local`** (subprocess), **`jupyter`** (remote kernel), or **`ssh`** (remote exec over SSH). |
| `max_wall_secs` | Upper bound on each run’s `timeout_secs` (default 300, max 86400). |
| `max_output_bytes` | Max combined stdout+stderr per run (default 256 KiB). |
| `max_sessions` | Max concurrent sessions (default 32). |
| `allowed_providers` | e.g. `["local"]`, `["jupyter"]`, `["ssh"]`; if empty or omitted, any implemented provider is allowed. |
| `python_executable` | Command for **local** Python runs and `execution_env_info` (default `python`). Ignored for Jupyter execution. For **SSH**, the remote interpreter is **`[harness.execution.ssh].remote_python`** (default `python3`). |
| `local_python_mode` | **`repl`** (default, or any value other than the opt-outs below): one **local** Python interpreter per session. **`subprocess`**, **`fresh`**, **`stateless`**, **`one_shot`**, **`false`**, **`0`**: each **`execution_run`** starts a new **`python -u -`** process (no shared namespace). |
| `artifact_max_file_bytes` | Max bytes per saved artifact file (default 4 MiB, clamped 64 KiB–64 MiB). |
| `artifact_max_total_bytes_per_run` | Max total bytes for all artifacts in one `execution_run` (default 32 MiB). |
| `artifact_max_files_per_run` | Max artifact files per run (default 64, clamped 1–256). |

Top-level **`doom_loop_enabled`** (optional, default **true**): when true, the agent detects repeated identical tool calls and injects a corrective user message before the next LLM call (see `src/agent/doom_loop.rs`).

Each successful **`execution_run`** also appends one JSON line to **`workspace_dir/.system_generated/execution_runs.jsonl`** (metadata only: no code body) and emits **`ExecutionRunFinished`** telemetry.

When `default_provider = "jupyter"`, add **`[harness.execution.jupyter]`**:

| Key | Meaning |
|-----|--------|
| `base_url` | Jupyter Server root, e.g. `http://127.0.0.1:8888` (no `/lab` path). **Required** for Jupyter. |
| `token` | Optional server token. Prefer host env **`JUPYTER_TOKEN`** (wins over this field) so secrets are not committed. |
| `kernel_name` | Kernel spec name for `POST /api/kernels` when `language` is Python or unset (default **`python3`**). |

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

Restart the agent after editing config.

## Workspace layout (important)

- **`workspace_dir`** (outer): holds `config.toml`, logs, `.system_generated/`, etc.
- **`sandbox_dir`** (inner): usually `workspace_dir/workspace` — this is where execution runs and where paths are resolved.

Filesystem tools and execution share the same **sandbox boundary** when `restrict_to_workspace = true` (default). Do not put secrets in the sandbox if the model can read them.

Materialized run artifacts live under **`sandbox_dir/.execution_artifacts/`** (session segment is sanitized for path safety). Operator scenarios: **`docs/execution-use-cases.md`**.

## Typical workflow (for you or the model)

1. **`execution_session_create`** — optional `label`, optional `language`.  
   - **`local`:** `python`, `py`, `shell`, `sh`, `bash`.  
   - **`jupyter`:** `python` / `py` / unset (uses `kernel_name`), or **`r`** / **`R`** (uses the **`ir`** kernel spec if installed).  
   - **`ssh`:** `python` / `py` / unset, or `shell` / `sh` / `bash`.  
   - Response includes **`session_id`** and capability summaries — keep the `session_id` for the next steps.

2. **`execution_run`** — required: `session_id`, `code`. Optional: `timeout_secs`, `cwd_mode` (`session_default` or `sandbox_relative`), and `cwd_relative` when using `sandbox_relative`.  
   - **`jupyter`:** only **`session_default`** is supported for `cwd_mode` (no per-run sandbox cwd); use notebook magics such as `%cd` inside `code` if you must change directory on the server.  
   - **`ssh`:** only **`session_default`** is supported; the remote working directory is always **`remote_workdir`** from config.

3. When finished (or to free slots): **`execution_session_close`** with the same `session_id`.

Use **`execution_cancel`** if a run is stuck and the provider reports **`supports_interrupt`** (true for **`local`** and **`jupyter`**; **false** for **`ssh`** in the current release).

## Python and virtual environments (local provider)

The **local** harness runs **`python_executable`** as a normal process. It does **not** auto-activate a venv; you should either:

- Set **`python_executable`** to the interpreter you want (e.g. path to `uv`-managed `.venv\Scripts\python.exe` on Windows, or `.../.venv/bin/python` on Unix), or  
- Rely on a shell session (`language: shell`) and invoke `uv run …` / activate scripts in **`code`** (understand the security tradeoff of shell mode).

If something fails, check **`execution_env_info`** and the tool error text (missing interpreter, timeout, etc.).

For **Jupyter**, pick the kernel environment by **`kernel_name`** and the kernels installed on that server; the agent does not configure `pip` or conda from tool args in this release.

**Notebook vs Lab:** both use **Jupyter Server**; the kernel WebSocket URL and message framing are the same. Use **`base_url`** as the server root (for example `http://127.0.0.1:8888`), not the `/lab?token=…` UI URL—put the token in **`JUPYTER_TOKEN`** or `[harness.execution.jupyter].token` instead.

**Output capture:** the server may send `print()` output as **JSON text** WebSocket frames or as **binary v1** frames (depending on subprotocol). The agent collects **`stream`** (stdout/stderr), **`execute_result`** / **`display_data`** (`text/plain`), the iopub **`error`** message (traceback text goes to stderr once — **`execute_reply`** with `status: error` does not duplicate it), and **`execute_reply`** for exit status. A run finishes only after **`execute_reply`** and an iopub **`status`** `execution_state: idle` with the same parent `msg_id`, so trailing **`stream`** / **`execute_result`** frames are not dropped early. If the socket closes first, the client returns best-effort output after **`execute_reply`**. Bare expressions (last line without `print`) appear via **`execute_result`**, not always as a `stream`.

The client requests WebSocket subprotocol **`v1.kernel.websocket.jupyter.org`** when opening `/api/kernels/{id}/channels` (Jupyter Server’s preferred binary layout). If the handshake fails, it retries **without** that header so older or unusual proxies still work.

### Jupyter: run a server locally (quick start)

1. In the same environment where you want the kernel (e.g. your project venv), install Jupyter if needed: `pip install jupyterlab` (or `notebook`).
2. Start the server without opening a browser, on a fixed port, for example:
   ```bash
   jupyter lab --no-browser --port=8888
   ```
3. Copy the **token** from the printed URL (or set a password in Jupyter config). On the **host** running isanagent, set:
   ```bash
   set JUPYTER_TOKEN=...your-token...
   ```
   (Unix: `export JUPYTER_TOKEN=...`) or put `token = "..."` under `[harness.execution.jupyter]` only for local experiments.
4. In **`config.toml`**, set `default_provider = "jupyter"` and:
   ```toml
   [harness.execution.jupyter]
   base_url = "http://127.0.0.1:8888"
   ```
   Use the **server root** (`http://…:port`), not the `/lab?token=…` page URL.
5. Restart isanagent. You may see Jupyter log **`No session ID specified`** on first WebSocket traffic; that is a known server warning and does not block execution.

### Jupyter: troubleshooting

| Symptom | Things to check |
|--------|------------------|
| `401` / `403` on REST or WS | Token: **`JUPYTER_TOKEN`** env vs `[harness.execution.jupyter].token`; URL must match the server you started. |
| `unknown kernel` / kernel start fails | **`kernel_name`** must match an installed kernelspec (`jupyter kernelspec list` on the server host). |
| Empty `stdout` on older builds | Upgrade to a build that handles **text JSON** and **v1 binary** server messages (see “Output capture” above). |
| Wrong Python / packages | The kernel uses the **server’s** environment, not the agent sandbox; install packages in that env or pick another kernelspec. |

## Working directory for a run

- **`cwd_mode`: `session_default`** (default) — run in the session’s root (sandbox root).  
- **`cwd_mode`: `sandbox_relative`** — requires **`cwd_relative`** (e.g. `pkg`); resolved under the sandbox like other tools.

## Sub-agents

If you use **`[harness.subagents]`** with **`allowed_tools`**, include the execution tool names explicitly if sub-agents should run code:

`execution_session_create`, `execution_run`, `execution_artifact_list`, `execution_cancel`, `execution_session_close`, `execution_env_info`

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

- **Implemented:** Jupyter provider (`execution-implementation-plan.md` Phase 3); SSH MVP (`execution-implementation-plan.md` Phase 4); Phase 6 artifacts, **`execution_artifact_list`**, run manifest (`execution_runs.jsonl`), telemetry **`ExecutionRunFinished`**, and **`doom_loop_enabled`**.  
- **Later:** Hosted / Colab-shaped provider (`execution-implementation-plan.md` Phase 5); execution provisioners (deferred design doc).

When we add providers or config keys, this guide and **`AGENTS.md`** should be updated in the same change so operators are not surprised.
