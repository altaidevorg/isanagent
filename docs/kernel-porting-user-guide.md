# MaxEvolve Kernel Porting — Operator Guide

MaxEvolve combines MaxKernel-style multi-agent Triton→Pallas translation with AlphaEvolve-style MAP-Elites evolution inside isanagent.

## Enable

In `config.toml`:

```toml
[harness.kernel_porting]
enabled = true

[harness.subagents]
enabled = true

[harness.execution]
enabled = true
local_python_runtime = "uv_managed"
```

Onboarding copies the `kernel-porting` skill, agent prompts, reference docs, and benchmarks.

## Tools

| Tool | Purpose |
|------|---------|
| `kernel_db_init` | Create `kernels/projects/{id}/` layout + empty MAP-Elites DB |
| `kernel_db_sample` | Top elites for mutation prompts |
| `kernel_db_insert` | Record profiled candidate |
| `kernel_db_status` | Archive summary / global best |

Sub-agents: `kernel_orchestrator`, `gpu_to_jax`, `implement_kernel`, `test_generator`, `test_runner`, `kernel_profiler`, `kernel_mutator`, `evolve_orchestrator`.

## Typical session

1. User provides Triton/PyTorch source path.
2. Coordinator loads `kernel-porting` skill → spawns `kernel_orchestrator`.
3. Orchestrator runs `kernel_db_init` and GpuToJax 12-step plan (`.agents/kernel-porting/gpu_to_jax_plan.json`).
4. Correctness gate: `test_correctness.py` via `correctness_check.py` with `JAX_PLATFORMS=cpu`.
5. Optional evolution: `evolve_orchestrator` + Colab/SSH profiling.
6. Deliver `REPORT.md` from template in skill.

## Validators

Located at `skills/kernel-porting/scripts/validators/`. Each script uses **PEP 723** inline metadata; run with **`uv run`** (workspace already uses UV):

```bash
# from sandbox root
uv run skills/kernel-porting/scripts/validators/correctness_check.py kernels/projects/{id}/test_correctness.py
```

Or invoke the same commands through `execution_run` / `exec`. Set `JAX_PLATFORMS=cpu` for local interpret/correctness.

Copy-paste test prompts: `kernels/Test_Prompts.md` (also under `kernels/reference/` after onboard).

## Hardware

| Environment | Mechanism |
|-------------|-----------|
| Local CPU interpret | `execution_run` (local provider) |
| Colab GPU/TPU | `colab-cli` skill + `exec` |
| Remote server | `[harness.execution.ssh]` |

Always `colab stop` after remote profiling.

## Benchmarks

Shipped under `workspace/benchmarks/`:

- `vector_add` — Phase 1 E2E
- `matmul_relu` — Phase 2 evolution
- `flash_attention`, `block_sparse_attention` — Phase 4

## Windows note

Build isanagent in **release** mode on Windows. Local JAX is CPU-only; use Colab or SSH for GPU/TPU MFU.

## See also

- [Execution user guide](execution-user-guide.md)
- `workspace/kernels/reference/` — conversion research docs
- Root `AGENTS.md` — MaxEvolve architecture summary
