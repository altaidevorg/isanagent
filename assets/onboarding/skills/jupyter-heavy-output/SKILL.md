---
name: jupyter-heavy-output
description: Interpret Jupyter execution sessions in isanagent—attachments, kernel cwd, tokens; contrast with local provider behavior.
requires:
  bins: []
  env: []
always: false
---

# Jupyter execution and heavy outputs (isanagent)

Apply this when **`execution_env_info`** (or the session you opened) shows the **Jupyter** provider—i.e. the host configured **`default_provider = "jupyter"`** and valid **`[harness.execution.jupyter]`** settings.

## What you can rely on from tools

- **`execution_session_create`** returns **`session_capabilities`** (JSON) for that session. Jupyter sessions use a **remote kernel** with **persistent variables** across `execution_run` calls until the session is closed or the kernel dies.
- **`execution_run`** may return **`attachments`**: sandbox-relative paths for materialized **`image/png`**, **`image/jpeg`**, large **`text/csv`**, large **`application/json`**, etc. Short human-readable lines may appear in stdout; **do not assume** full binary or huge JSON appears in the tool string—open paths via **`execution_artifact_list`** + file tools, and describe outcomes in prose.

## Paths and working directory

- Jupyter **`execution_run`** only supports **`cwd_mode: session_default`** in practice—the process cwd is on the **Jupyter server**, not your agent sandbox. If the user needs a server-side directory change, encode it in code (e.g. **`os.chdir`** / **`%cd`**) explicitly; do not confuse server paths with **`sandbox_relative`** cwd for the **local** provider.

## Local vs Jupyter (mental model)

| Concern | Local provider | Jupyter provider |
|--------|----------------|------------------|
| Where code runs | Child / REPL worker on agent host, cwd under **sandbox** | Remote kernel on Jupyter server |
| Packages | Same as agent host `python` | Whatever is installed **on the server** |
| Big plot/table | Prefer write under **`.execution_artifacts/...`** or attachments pattern | Often **`attachments`** from `display_data` |
| Interrupt | **`execution_cancel`** + harness behavior | Uses server interrupt when supported |

Check **`execution_session_create`** metadata (e.g. `extensions.local_python_repl` on local) when you need to remember whether **local Python** is REPL-style or subprocess—do not guess from memory.

## Secrets

Prefer the **`JUPYTER_TOKEN`** environment variable on the agent process over pasting tokens into replies. Never echo tokens into chat or logs.
