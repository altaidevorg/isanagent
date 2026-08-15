---
name: kernel-porting
description: MaxEvolve Triton/PyTorch to JAX/Pallas porting with GpuToJax 12-step pipeline, interpret=True validation, MAP-Elites evolution, and Colab/SSH hardware profiling.
requires:
  bins: ["uv"]
  env: []
always: false
---

# MaxEvolve Kernel Porting

Native isanagent workflow combining **MaxKernel-style** multi-agent translation with **AlphaEvolve-style** MAP-Elites search.

## When to load

User mentions Triton, Pallas, custom kernels, JAX porting, or MaxEvolve. Delegate to `kernel_orchestrator` via `subagent_spawn(agent="kernel_orchestrator", ...)`.

## Hard rules (never violate)

1. **No pointer arithmetic** inside Pallas kernel bodies — use `BlockSpec` + `pallas_call` orchestration only.
2. **`jax.custom_vjp`** for production backward passes — do not rely on bare `jax.grad` for optimized kernels.
3. **`interpret=True`** CPU gate before any hardware run.
4. Block-sparse / dynamic fetch → `PrefetchScalarGridSpec` (see `kernels/reference/Triton To Pallas Conversion.md`).

## Project layout

```
kernels/projects/{project_id}/
  source/                 # input Triton/PyTorch
  SIMPLIFICATION_PLAN.md
  organized_gpu.py
  converted_jax.py
  test_correctness.py
  profile_script.py
  database/map_elites.json
  database/lineage.jsonl
  candidates/queue.jsonl
  artifacts/
  REPORT.md
```

Initialize with `kernel_db_init(project_id=..., target_hardware=cpu_interpret|gpu_hopper|tpu_v5e|tpu_v6e)`.

## Five-phase workflow

1. **Init** — `kernel_db_init`, `todo_write`, copy source into project.
2. **GpuToJax 12-step** — `subagent_plan_execute` using template `.agents/kernel-porting/gpu_to_jax_plan.json` (each step `agent: gpu_to_jax`). HITL: `ask_user` after step 2 (plan approval).
3. **Verification** — `test_generator` then `test_runner`; validators via `execution_run` or `exec`.
4. **Evolution** — `evolve_orchestrator` + `kernel_mutator`; `kernel_db_sample` / `kernel_db_insert`; background `evolve_runner.py`.
5. **Delivery** — copy global best to `best_kernel.py`, write `REPORT.md`.

## Validators (`uv run` + PEP 723)

All scripts under `skills/kernel-porting/scripts/` declare dependencies inline (PEP 723). Run with **`uv run`** — the workspace already uses UV via `[harness.execution] local_python_runtime = "uv_managed"`.

Set `JAX_PLATFORMS=cpu` for local interpret/correctness.

**From project directory** (`execution_run` with `cwd_relative=kernels/projects/{id}`):

```bash
uv run ../../skills/kernel-porting/scripts/validators/jax_syntax_check.py converted_jax.py
uv run ../../skills/kernel-porting/scripts/validators/compile_check.py converted_jax.py
uv run ../../skills/kernel-porting/scripts/validators/shape_check.py converted_jax.py
uv run ../../skills/kernel-porting/scripts/validators/correctness_check.py test_correctness.py
```

**From sandbox root:**

```bash
uv run skills/kernel-porting/scripts/validators/jax_syntax_check.py kernels/projects/{id}/converted_jax.py
uv run skills/kernel-porting/scripts/validators/correctness_check.py kernels/projects/{id}/test_correctness.py
```

Generated project artifacts (`converted_jax.py`, `test_correctness.py`, `profile_script.py`) should also use PEP 723 metadata so `uv run converted_jax.py` works standalone.

## Hardware routing

| Target | Path |
|--------|------|
| Correctness / interpret | `execution_run` local provider + `uv run` validators |
| GPU/TPU benchmarks | `load_skill_instructions(colab-cli)` → `colab start` → `colab exec -f profile_script.py` → **`colab stop`** |
| Persistent remote | `[harness.execution.ssh]` + `execution_run_background` |

Profile locally (sandbox root):

```bash
uv run skills/kernel-porting/scripts/profile/roofline_mfu.py kernels/projects/{id}/profile_script.py
```

Windows hosts: local JAX is CPU-only; use Colab or SSH for GPU/TPU MFU.

## Evolution

- Sample parents: `kernel_db_sample(project_id, top_k=3)`
- Mutator writes `candidates/{uuid}/converted_jax.py`, append queue line to `candidates/queue.jsonl`
- Batch process (background job, sandbox root):

```bash
uv run skills/kernel-porting/scripts/evolve/evolve_runner.py --project {id} --generations N
```

- Record elite: `kernel_db_insert` with `fitness_latency_ms` (required), optional `fitness_mfu`, `mutation_class`

Mutation classes: `tiling`, `pipelining`, `layout`, `sparsity` (see `scripts/evolve/mutation_spec.json`).

## Reference docs

- `kernels/reference/Triton To Pallas Conversion.md`
- `kernels/reference/Implementation_Plan.md`
- `kernels/Test_Prompts.md` — copy-paste prompts to exercise MaxEvolve

## Dependencies

Workspace `[harness.execution] uv_requirements` also lists `jax`, `jaxlib`, `pytest`, `numpy` for the execution REPL. Validator scripts carry their own PEP 723 metadata so **`uv run path/to/script.py`** is self-contained.
