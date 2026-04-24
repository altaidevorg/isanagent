---
name: long-running-execution
description: Configure long wall clocks and use execution_run_background + job polling for ML or hours-long runs without blocking the agent loop.
requires:
  bins: []
  env: []
always: false
---

# Long-running and background execution

## Config (operator)

1. Set **`[harness.execution] max_wall_secs`** to the upper bound you need (clamped **1–86400**). The default **300** is only a default—not a hard ceiling in code.
2. Optionally set **`default_execution_timeout_secs`** so omitted **`timeout_secs`** on **`execution_run`** / **`execution_run_background`** defaults to something sensible (still clamped to **`max_wall_secs`**).
3. For each call, pass **`timeout_secs`** up to that cap when you want an explicit limit.

## Agent pattern

1. **`execution_session_create`** as usual.
2. Start work with **`execution_run_background`** (same args as **`execution_run`**, plus optional **`label`**). You receive **`job_id`** immediately.
3. Poll **`execution_job_status`** until **`terminal`** is true (or use channel/UI notices if present).
4. Fetch output with **`execution_job_result`** ( **`RunResult`** JSON; may truncate to **`max_tool_output_chars`** ).
5. Interrupt with **`execution_job_cancel`** or **`execution_cancel`** on the session when **`supports_interrupt`** is true.

## Constraints

- **One** active run or background job per **`session_id`**. Parallel long jobs need **separate sessions** (or a provider that allows more).
- Jobs live **in the agent process**; they are gone after exit. For survive-restart orchestration, use an external scheduler (Slurm, K8s, supervised scripts)—out of scope for in-process jobs.
- Write **checkpoints** and logs under the workspace so work survives cancellation or process loss.
