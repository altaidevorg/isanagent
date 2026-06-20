# MaxEvolve Test Prompts for isanagent

Copy-paste these into the terminal TUI or API chat. See `docs/kernel-porting-user-guide.md` for setup.

## Prerequisites

Your workspace `config.toml` must include:

```toml
[harness.kernel_porting]
enabled = true

[harness.subagents]
enabled = true

[harness.execution]
enabled = true
local_python_runtime = "uv_managed"
```

After `isanagent onboard`, confirm these exist:

- `workspace/skills/kernel-porting/SKILL.md`
- `workspace/benchmarks/vector_add/source/vector_add_triton.py`
- `workspace/.agents/prompts/kernel_orchestrator.md`

Run isanagent in **release** mode on Windows.

Validators and helpers use **`uv run`** with PEP 723 inline dependencies (no separate `pip install` step when `uv` is on PATH — already true for `[harness.execution] uv_managed`).

---

## 1. Smoke test (DB init only)

```
Load the kernel-porting skill, then spawn kernel_orchestrator with wait=true and this task:

Initialize a MaxEvolve project:
- project_id: vector_add_v1
- target_hardware: cpu_interpret
- source_relative_path: benchmarks/vector_add/source/vector_add_triton.py

Use kernel_db_init, then kernel_db_status and report the project layout under kernels/projects/vector_add_v1/.
Do not start the 12-step conversion yet.
```

**Success:** `kernel_db_init` returns `"status": "initialized"`; `kernels/projects/vector_add_v1/database/map_elites.json` exists.

---

## 2. Full Phase 1 test (recommended)

```
I want to test MaxEvolve end-to-end on the shipped vector_add benchmark.

1. Load the kernel-porting skill instructions.
2. Spawn agent=kernel_orchestrator with wait=true and delegate the full workflow:
   - kernel_db_init(project_id=vector_add_v1, target_hardware=cpu_interpret,
     source_relative_path=benchmarks/vector_add/source/vector_add_triton.py)
   - Run the 12-step GpuToJax pipeline using subagent_plan_execute and the plan at
     .agents/kernel-porting/gpu_to_jax_plan.json (each step should use agent gpu_to_jax).
   - After step 2 (SIMPLIFICATION_PLAN.md), use ask_user for my approval before continuing.
   - Run validators via execution_run using uv run (see kernel-porting skill), with
     cwd_relative=kernels/projects/vector_add_v1 and JAX_PLATFORMS=cpu in the environment.
   - Stop after correctness passes; skip evolution and Colab for now.
3. Summarize: paths to converted_jax.py, test_correctness.py, and validator JSON output.

Constraints:
- No pointer arithmetic inside the Pallas kernel body; use BlockSpec orchestration.
- Use interpret=True / CPU pytest for correctness.
- Do not skip validator gates.
```

When prompted for plan approval:

```
Plan approved. Proceed with steps 3–12 and run the correctness gate.
```

**Success:**

- `kernels/projects/vector_add_v1/converted_jax.py`
- `kernels/projects/vector_add_v1/test_correctness.py`
- `correctness_check` validator returns `"ok": true`
- Optional reference: `benchmarks/vector_add/example/converted_jax_reference.py`

---

## 3. Short natural-language prompt

```
Port the Triton vector-add kernel in benchmarks/vector_add/source/vector_add_triton.py
to JAX Pallas using MaxEvolve. Target CPU interpret validation only for now.
Use the kernel-porting skill and kernel_orchestrator. Project id: vector_add_v1.
Ask me to approve the simplification plan before implementation.
```

---

## 4. Manual validator commands (sandbox shell or execution_run)

From project directory (`cwd_relative=kernels/projects/vector_add_v1`):

```bash
set JAX_PLATFORMS=cpu
uv run ../../skills/kernel-porting/scripts/validators/jax_syntax_check.py converted_jax.py
uv run ../../skills/kernel-porting/scripts/validators/compile_check.py converted_jax.py
uv run ../../skills/kernel-porting/scripts/validators/shape_check.py converted_jax.py
uv run ../../skills/kernel-porting/scripts/validators/correctness_check.py test_correctness.py
```

From sandbox root:

```bash
set JAX_PLATFORMS=cpu
uv run skills/kernel-porting/scripts/validators/jax_syntax_check.py kernels/projects/vector_add_v1/converted_jax.py
uv run skills/kernel-porting/scripts/validators/correctness_check.py kernels/projects/vector_add_v1/test_correctness.py
```

---

## 5. Evolution smoke test (optional)

Only after Phase 1 passes. Colab optional.

```
MaxEvolve evolution test on vector_add_v1 (assume conversion already passes correctness).

Spawn evolve_orchestrator with wait=true:
- kernel_db_sample(project_id=vector_add_v1, top_k=3)
- One kernel_mutator mutation (tiling class only): write candidate under candidates/
- test_runner correctness filter
- Skip Colab unless I confirm; if local only, insert a synthetic elite with kernel_db_insert
  (fitness_latency_ms=1.0, notes=smoke test) and report kernel_db_status.

Do not run more than 1 generation.
```

Background batch (orchestrator may run this via `execution_run_background` from sandbox root):

```bash
uv run skills/kernel-porting/scripts/evolve/evolve_runner.py --project vector_add_v1 --generations 1
```

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| No `kernel_orchestrator` | `[harness.kernel_porting] enabled`, `[agents.kernel_orchestrator]` in config |
| `kernel_db_*` tool missing | Rebuild isanagent; enable kernel_porting harness |
| JAX import errors | `uv` on PATH; run from workspace with onboarding `uv_requirements` |
| Agent skips MaxEvolve | Mention Triton/Pallas/MaxEvolve explicitly or use prompt **2** |
