# Execution harness — scripted use cases

Short scenarios for manual or regression testing when the execution harness is active (it is on by default unless **`[harness.execution] enabled = false`**). All paths are relative to your workspace sandbox unless noted.

## 1. Local Python smoke (system runtime)

1. `execution_session_create` with `language: python` (or `shell` on Windows if that is your default test path).
2. `execution_run` with `print("ok")` and a small `timeout_secs`.
3. `execution_session_close`.

Expect: JSON `RunResult` with `exit_code` 0, stdout containing `ok`, `attachments` usually empty.

## 2. Local UV-managed Python smoke

Requires:
- `default_provider = "local"`
- `[harness.execution] local_python_runtime = "uv_managed"`
- `uv` installed on host `PATH`

1. `execution_session_create` with `language: python`.
2. `execution_run` with `import sys; print(sys.executable)`.
3. Verify stdout contains the managed env path under `.system_generated/uv/envs/`.
4. Close and create a second session; run the same code and verify the same interpreter path is reused.
5. `execution_session_close`.

Expect: first run may take longer due to `uv venv` setup; second run reuses cached env.

## 3. Jupyter plot and artifacts

Requires a running Jupyter Server and `default_provider = "jupyter"` plus **`JUPYTER_TOKEN`**.

1. `execution_session_create` (Python kernel).
2. `execution_run` with code that displays a figure, e.g. matplotlib `plt.figure(); plt.plot([0,1]); plt.show()` or equivalent that emits **`display_data`** with PNG.
3. Inspect `RunResult.attachments` for sandbox-relative paths under `.execution_artifacts/…`.
4. `execution_artifact_list` with the same `session_id`.
5. `read_file` is appropriate only for text artifacts; open binary PNGs outside the agent if needed.

## 4. Interrupt (local or Jupyter)

1. Create session, then `execution_run` with a long sleep/busy loop and a high `timeout_secs`.
2. While running, call **`execution_cancel`** (supported for local and Jupyter, not SSH MVP).

Expect: run ends with cancel or timeout semantics per provider.

## 5. SSH remote exec (manual)

With **`default_provider = "ssh"`** and valid **`[harness.execution.ssh]`** credentials:

1. `execution_session_create`, `execution_run` with trivial Python on the remote, `execution_session_close`.

Expect: stdout from remote; **`execution_cancel`** does not stop the remote process (client wait only).

## 6. Manifest and telemetry

After any successful **`execution_run`**, check **`workspace_dir/.system_generated/execution_runs.jsonl`** for a new JSON line (timestamp, `chat_id` when invoked from a real chat, `provider_id`, lengths, `artifact_count`, optional `git_head`). Structured logs may also record **`ExecutionRunFinished`** telemetry.

## 7. Doom loop correction (agent behavior)

With **`doom_loop_enabled = true`** (default) at the top level of **`config.toml`**, if the model issues the same tool with identical arguments three times in a row, the agent injects a corrective **user** message into history before the next LLM call. Verify in logs: “Doom loop detected”.
