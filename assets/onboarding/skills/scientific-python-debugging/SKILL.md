---
name: scientific-python-debugging
description: Debug failing scientific Python inside isanagent execution_run—tracebacks, truncation, seeds, provider mismatch.
requires:
  bins: []
  env: []
always: false
---

# Scientific Python debugging (isanagent execution harness)

Use this when **`execution_run`** returns non‑zero exit, empty stdout, traceback text, or results that look nondeterministic or environment‑wrong.

## 1. Read the last tool output carefully

- **Stderr** usually carries the traceback. If stderr mentions truncation (`... (truncated)`), you hit **`max_output_bytes`**—rerun with less logging, write logs to a file under the sandbox, or **`tail`-style** smaller prints.
- **Exit code** on local Python REPL reflects the worker’s status flag (0 = clean, 1 = exception caught in the worker). On subprocess / shell it is the real process exit code.

## 2. Reproduce smaller

Shrink data (fewer rows, smaller tensor, subset of files) until the failure is obvious. Set **explicit random seeds** (`random`, **`numpy.random.seed`** / **`Generator`**, framework-specific seed args) so “flakey” runs become stable.

## 3. Confirm environment with **`execution_env_info`**

- **Local:** packages = host interpreter you run under.
- **Jupyter:** packages = **server** kernel environment—imports that work locally may fail there. Adjust install instructions for the user on the **server**, not in the agent sandbox.

## 4. Separate “code bug” from “resource bug”

- **Hang / no output:** try **`execution_cancel`**, then shorter **`timeout_secs`**, smaller input, or progress prints to a file.
- **`NameError` / `ImportError` after an earlier success:** if local REPL mode is on, you may have shadowed a name or partially executed cells—reread prior runs in the same session or restart by **`execution_session_close`** + new session.
- **`FileNotFoundError`:** paths must be valid **inside the sandbox** (or server cwd for Jupyter)—use **`glob_files`** / **`list_directory`** before assuming paths.

## 5. When to stop iterating

After **two** substantially different fixes still fail, summarize: hypothesis, evidence (stderr snippet, exit code), and what you already tried. Call **`ask_user`** only if you need a decision you cannot infer (e.g. which remote dataset, credentials, or kernel image).

## References (for you, not the end user)

[Python — errors and exceptions](https://docs.python.org/3/tutorial/errors.html), [traceback](https://docs.python.org/3/library/traceback.html).
