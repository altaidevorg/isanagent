---
name: scientific-python-debugging
description: Compact checklist for debugging scientific Python runs inside execution_session / execution_run.
requires:
  bins: []
  env: []
always: false
---

# Scientific Python debugging (execution harness)

1. **Reproduce minimally** — shrink data and code until the failure is obvious; fix randomness with explicit seeds where relevant ([Python `random` module](https://docs.python.org/3/library/random.html), NumPy seeds).
2. **Read the traceback** — stderr from **`execution_run`** usually contains the full exception; do not ignore partial output when `max_output_bytes` truncates; rerun with less logging or write trace to a file under the sandbox.
3. **Environment** — call **`execution_env_info`**; for Jupyter, the kernel’s packages come from the **server** environment, not the agent host.
4. **Artifacts** — for large diagnostics, write to `.execution_artifacts/...` or rely on Jupyter attachments instead of printing huge blobs.
5. **Escalate** — if stuck after two different approaches, summarize hypothesis and failed attempts; use **`ask_user`** only when human input is truly required.

Official references: [Python tutorial — errors](https://docs.python.org/3/tutorial/errors.html), [traceback module](https://docs.python.org/3/library/traceback.html).
