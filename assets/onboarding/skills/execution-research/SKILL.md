---
name: execution-research
description: Safe workflow for running analysis code, saving plots/tables as artifacts, and inspecting outputs without blowing the LLM context window.
requires:
  bins: []
  env: []
always: false
---

# Execution research workflow

Use this when the user wants reproducible numeric or ML work inside the workspace.

1. Call **`execution_env_info`** once to confirm provider, caps, and artifact limits.
2. **`execution_session_create`** with the right `language` (Python locally/Jupyter/SSH per config).
3. Prefer **small, incremental** `execution_run` steps. After plots or large tables, rely on **`RunResult.attachments`** (Jupyter) or explicit `savefig` / file writes under **`.execution_artifacts/<session_id>/`** (local) so outputs stay on disk.
4. Use **`execution_artifact_list`** with the `session_id` to enumerate saved files, then **`glob_files`** / **`search_text`** on text logs or CSV previews as needed.
5. **`execution_session_close`** when done to free the session slot.

Do not paste multi-megabyte base64 into the chat; always reference sandbox-relative paths returned by tools.
