---
name: ml-execution-preflight
description: Checklist before long execution_run or execution_run_background for ML/training scripts.
always: true
---

# ML execution preflight

Run these checks **before** a long or GPU-heavy `execution_run` / `execution_run_background`:

1. **`execution_env_info`** — note `max_wall_secs`, default timeout, provider id, and artifact caps.
2. **Interpreter / kernel** — for Jupyter or SSH, run a one-line `execution_run` that prints Python version and (if relevant) `torch.cuda.is_available()` or equivalent.
3. **Timeouts** — set `timeout_secs` explicitly from model size and hardware; training almost always needs a large value within `max_wall_secs`.
4. **Pilot** — run the smallest slice that still exercises the code path (few steps / tiny data), then scale.
5. **Logging** — prefer plain-text step logs the model can read from captured stdout (avoid hiding loss inside unreadable progress noise).
6. **Outputs** — decide where checkpoints and metrics will live (`execution_artifact_list`, sandbox paths, or copy-out plan). Ephemeral sandboxes lose data when torn down.
7. **Single path first** — do not launch many parallel heavy jobs until one configuration completes successfully.

If any step fails, fix the root cause before scaling.

**Before launching a fine-tuning or dataset generation script, also load `long-running-execution` with `load_skill_instructions`.