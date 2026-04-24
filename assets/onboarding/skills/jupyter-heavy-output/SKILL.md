---
name: jupyter-heavy-output
description: When to use Jupyter vs local execution; kernel cwd; binary display_data vs stdout.
requires:
  bins: []
  env: []
always: false
---

# Jupyter and heavy outputs

## When to prefer Jupyter

Use **`default_provider = jupyter`** when the user needs **persistent kernel state** (variables across runs), **interruptible** long cells, or rich **`display_data`** (plots). Local subprocess mode starts fresh each run unless you use a dedicated REPL pattern.

## Paths and cwd

Jupyter runs only support **`cwd_mode: session_default`**. The kernel’s working directory is on the **server**, not the agent sandbox. Use **`%cd`** in code if you must change directory there.

## Binary and large payloads

For **`image/png`**, **`image/jpeg`**, large **`text/csv`**, and large **`application/json`**, the agent may **materialize files** under:

`.execution_artifacts/<sanitized_session_id>/<run_uuid>/`

The **`execution_run`** JSON includes **`attachments`** with sandbox-relative paths and MIME hints. Short summaries may appear in stdout; do not expect full binary in the tool return.

Use **`execution_artifact_list`** to browse. Use **`read_file`** only for text; open binary images with external tools if needed.

## Tokens

Prefer **`JUPYTER_TOKEN`** in the process environment over committing tokens in `config.toml`.
