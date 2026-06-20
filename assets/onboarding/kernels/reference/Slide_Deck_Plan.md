# Slide Deck Plan: From Triton to Pallas

**Talk title:** From Triton to Pallas: Agentic, Evolutionary Kernel Porting from PyTorch to JAX  
**Implementation reference:** native **MaxEvolve** in [isanagent](https://github.com/Yusuf/agent-rs) — not a MaxKernel/ADK fork. Operator guide: `docs/kernel-porting-user-guide.md`.

**Deck structure:** 16 slides (expanded from 13). Slides 1–4 set motivation and fundamentals; 5–6 cover manual porting pain and the agentic paradigm; 7–12 cover MaxEvolve design; 13–15 cover validation, demo, and failure modes; 16 closes.

---

## Slide 1: Title and Abstract

* **Slide Title:** From Triton to Pallas: Agentic, Evolutionary Kernel Porting from PyTorch to JAX
* **Subtitle:** MaxEvolve — closed-loop multi-agent porting + MAP-Elites search inside isanagent
* **Visual Layout & Diagram Description:**
  * **Layout:** Three-panel hero slide. **Left:** PyTorch/Triton logo motif + imperative pointer-arithmetic snippet (faded). **Center:** isanagent workspace icon with branching sub-agent nodes and a circular evolution loop. **Right:** JAX/Pallas logo motif + BlockSpec orchestration snippet (faded). A DNA helix wraps the center panel only (evolution, not translation alone).
  * **Mermaid (optional inline graphic):**
    ```mermaid
    flowchart LR
      Triton[Triton / PyTorch] --> MaxEvolve[MaxEvolve in isanagent]
      MaxEvolve --> Pallas[JAX / Pallas]
      MaxEvolve --> Evolve[MAP-Elites loop]
    ```
  * **AI Image Prompt:** "Minimalist dark presentation title slide. Left third: glowing blue neural network and CUDA/Triton aesthetic with tiny code showing pointer offsets. Right third: glowing purple JAX array grid with BlockSpec labels. Center: a holographic command center labeled 'MaxEvolve / isanagent' with small agent avatars orbiting a green DNA double helix. Subtitle text area at bottom. Cinematic, 16:9, no clutter."
* **Key Bullet Points:**
  * **Problem:** Porting custom kernels across ecosystems is slow, error-prone, and requires rare expertise (memory hierarchy + two DSL philosophies).
  * **Approach:** **MaxEvolve** — MaxKernel-*inspired* multi-agent workflow + AlphaEvolve-*inspired* MAP-Elites search, implemented **natively** as isanagent named agents, skills, Rust tools, and sandbox Python validators.
  * **Deliverable:** A reproducible pipeline from `benchmarks/vector_add/source/vector_add_triton.py` → `kernels/projects/{id}/converted_jax.py` → optional hardware-tuned elites in `database/map_elites.json`.
  * **Honest scope:** External headline numbers (Gemini +23%, Deepseek MLA −8.7% latency) come from Google/AlphaEvolve literature — our repo ships the **machinery** to pursue similar gains, not pre-baked production kernels.
* **Speaker Notes:**
  > Welcome. This talk is about treating kernel porting as an **engineering search problem**, not a one-shot LLM translation. We built **MaxEvolve** directly into **isanagent**: named sub-agents, a `kernel-porting` skill, MAP-Elites database tools, and Python validators you can run today. I'll separate what we **implemented** from what the literature **reports** as achievable once the loop runs on real silicon.

---

## Slide 2: Why Custom Kernels Still Matter

* **Slide Title:** Why We Still Write Custom Kernels in 2026
* **Visual Layout & Diagram Description:**
  * **Layout:** Top row = three workload cards (dense matmul, MoE routing, block-sparse attention). Bottom = single compiler path vs custom-kernel path converging on hardware.
  * **Diagram description:** Draw a **decision tree**. Root: "Is memory access regular and dense?" → Yes → "XLA / TorchInductor often sufficient." → No → "Custom kernel (Triton / Pallas) required." Highlight irregular branches in orange.
  * **AI Image Prompt:** "Infographic, dark blueprint style. Three panels top: (1) dense matrix multiply grid, green checkmark 'compiler OK'; (2) Mixture-of-Experts router with scattered experts, red X 'compiler struggles'; (3) block-sparse attention pattern, red X. Bottom: funnel from high-level PyTorch/JAX ops down to either 'Compiler autotune' or 'Hand-written / agent-written kernel'. Neon orange highlights on irregular paths."
* **Key Bullet Points:**
  * LLM training/inference bottlenecks increasingly come from **irregular** patterns: MoE, block-sparse attention, ragged batches, dynamic sequence lengths.
  * Compilers optimize **dense, static** graphs well; they lose arithmetic intensity when routing, masking, or sparsity forces repeated HBM round-trips.
  * Custom kernels exist to **orchestrate on-chip SRAM/VMEM** and overlap DMA with compute — not to rewrite `matmul` for fun.
  * **Talk hook:** Every minute saved on a hot kernel compounds across trillion-token training runs.
* **Speaker Notes:**
  > Start with *why* before *how*. The audience should leave understanding that kernel work is not nostalgia — it's what you do when the compiler safety net stops matching the model architecture.

---

## Slide 3: The Memory Wall and Roofline Intuition

* **Slide Title:** The Memory Wall: When FLOPs Are Not the Bottleneck
* **Visual Layout & Diagram Description:**
  * **Layout:** Left = physical metaphor (slow HBM conveyor → fast MXU/Tensor Core robot arm with STALL indicator). Right = roofline chart with a dot labeled "your kernel" sliding left/right as arithmetic intensity changes.
  * **Annotated roofline:** Draw horizontal **memory bandwidth roof** and sloped **compute roof**; mark TPU v5e ridge point ≈ **240 FLOPs/byte** (cite as rule-of-thumb, not universal).
  * **AI Image Prompt:** "Split technical slide. Left: industrial metaphor — wide slow conveyor belt labeled HBM feeding a tiny ultra-fast robotic MXU arm frozen with yellow STALL light. Right: classic roofline log-log plot, ridge point annotated '240 FLOPs/byte (TPU v5e class)'. A movable dot labeled 'Kernel under test' sits in memory-bound region. Dark theme, white grid, orange accent on stall."
* **Key Bullet Points:**
  * **Arithmetic intensity:** AI = FLOPs / bytes moved from HBM.
  * Below the ridge: **memory-bound** — faster math units do not help.
  * Above the ridge: **compute-bound** — tile size, pipelining, and layout matter.
  * MaxEvolve's evolutionary fitness targets **latency_ms**, **MFU**, and **TFLOPS** (via `kernel_db_insert` and `roofline_mfu.py`) — not LOC alone.
* **Speaker Notes:**
  > This slide grounds the evolutionary objective function. We are not asking the LLM to write pretty code; we ask whether the kernel **moves the roofline dot**.

---

## Slide 4: Triton vs. Pallas — Two Philosophies

* **Slide Title:** Triton vs. Pallas: Imperative Pointers vs. Declarative BlockSpec
* **Visual Layout & Diagram Description:**
  * **Layout:** Two-column **layer cake** diagram.
    * **Triton stack:** Python kernel → Triton-IR → TTGIR (layouts) → LLVM → PTX. Kernel body contains **pointer math**.
    * **Pallas stack:** Python kernel → JAX trace → jaxpr → StableHLO → Mosaic (TPU) / Mosaic GPU. Kernel body uses **Ref**; **BlockSpec** lives in orchestration layer.
  * **Center warning banner:** "Naive pointer-style code inside Pallas → 5–10× regression (Triton 3.x gather/scalar-load path)."
  * **AI Image Prompt:** "Side-by-side compiler stack diagrams. Left column header TRITON: layers pointing down to PTX, kernel snippet with tl.load and mask highlighted orange. Right column header PALLAS: layers down to Mosaic, kernel snippet with x_ref[...] and separate BlockSpec box highlighted green. Red warning banner between them about performance regression. Developer presentation aesthetic."
* **Key Bullet Points:**
  * **Triton:** `tl.program_id`, `tl.arange`, masks, explicit loads — imperative SPMD blocks.
  * **Pallas:** `pallas_call` + `grid` + `BlockSpec(index_map=..., block_shape=...)` — DMA planned before kernel math runs.
  * **Translation trap:** Copying Triton offsets into Pallas defeats coalescing/TMA-style block loads.
  * **MaxEvolve rule (enforced in skill + agent prompts):** *No pointer arithmetic in the kernel body.*
* **Speaker Notes:**
  > This is the core paradigm slide. Everything in our agent prompts and validators assumes this separation.

---

## Slide 5: What Breaks in Manual Porting

* **Slide Title:** Manual Triton→Pallas Porting: A Taxonomy of Pain
* **Visual Layout & Diagram Description:**
  * **Layout:** **Failure mode matrix** — rows = failure types, columns = detection stage.
  | Stage | Syntax | Shapes | Correctness | Performance | Autodiff |
  |-------|--------|--------|-------------|-------------|----------|
  | Manual dev | hours | days | silent wrong | profile blind | grad surprises |
  | MaxEvolve | validator scripts | shape_check | interpret=True pytest | Colab/SSH profile | custom_vjp prompt |
  * **Visual:** Five icons along a broken conveyor belt (Syntax error, OOB NaN, Slow scalar loads, Wrong gradient, Wrong tile). MaxEvolve agents patch each with a wrench labeled with tool names.
  * **AI Image Prompt:** "Dark infographic: five broken gears on a conveyor belt labeled 'Manual porting pipeline'. Each gear has an icon: syntax error, NaN memory grid, snail for slow loads, tangled arrows for bad gradients, wrong tile size. Below, five glowing wrenches labeled jax_syntax_check, shape_check, correctness_check, kernel_profiler, custom_vjp. Clean corporate-tech style."
* **Key Bullet Points:**
  * **Paradigm mismatch:** Engineers rewrite syntax instead of redesigning BlockSpec — works on CPU, dies on silicon.
  * **Validation gap:** Without `interpret=True`, OOB and races surface only after expensive compiles.
  * **Autodiff paradox:** `jax.grad` transposes memory patterns; optimized forward ≠ optimized backward.
  * **Search gap:** Hand tuning explores a handful of tile sizes; production needs **hundreds** of compile-profile cycles.
  * **Our response:** Structured 12-step pipeline + automated validators + MAP-Elites — not a single chat prompt.
* **Speaker Notes:**
  > Motivate agents as **process compression**, not magic. Manual porting fails at different layers; MaxEvolve assigns an agent or script to each layer.

---

## Slide 6: From Copilot to Closed-Loop Agentic Development

* **Slide Title:** Agentic Kernel Development: Beyond One-Shot Codegen
* **Visual Layout & Diagram Description:**
  * **Layout:** Three maturity levels as ascending stairs:
    1. **Copilot** — human writes, LLM suggests snippets (open loop).
    2. **Agent** — tool-using loop with files, tests, fixed prompt (partial loop).
    3. **MaxEvolve** — multi-agent roles + evolutionary database + hardware fitness (closed loop).
  * **Closed-loop diagram:** `Plan → Implement → Validate → Profile → Mutate → Database → (repeat)` with human HITL gate only at plan approval (`ask_user` after step 2).
  * **AI Image Prompt:** "Three-step staircase infographic labeled Copilot, Agent, MaxEvolve left to right ascending. Top of stairs: circular flywheel with arrows Plan, Implement, Validate, Profile, Mutate, DB. Small human icon at 'Plan approval' gate only. Futuristic dark UI, cyan and purple gradients."
* **Key Bullet Points:**
  * **MaxKernel (Google, reference design):** hierarchical agents for plan / implement / test / profile / autotune — we **reimplemented the roles**, not the ADK runtime.
  * **AlphaEvolve (DeepMind, reference design):** MAP-Elites + LLM mutations — we ship `kernel_db_*` + `evolve_runner.py` + `kernel_mutator` agent.
  * **isanagent contribution:** actor bus, sub-agents, execution harness, Colab skill — production-grade orchestration shell.
  * **Design choice:** Rust owns safety and persistence; Python owns JAX compile/test/evolve math.
* **Speaker Notes:**
  > Position the talk: we stand on MaxKernel/AlphaEvolve ideas but the **artifact is isanagent config + code you can clone today**.

---

## Slide 7: MaxEvolve Agent Map (What We Actually Ship)

* **Slide Title:** MaxEvolve on isanagent: Named Agents and Tools
* **Visual Layout & Diagram Description:**
  * **Layout:** Org chart rooted at coordinator → `kernel_orchestrator` sub-agent.
  * **Mermaid (for slide graphic or appendix):**
    ```mermaid
    flowchart TB
      User[User / main agent] --> KO[kernel_orchestrator]
      KO --> PK[plan_kernel]
      KO --> GJ[gpu_to_jax x12 steps]
      KO --> TG[test_generator]
      KO --> TR[test_runner]
      KO --> EO[evolve_orchestrator]
      EO --> KM[kernel_mutator]
      EO --> KP[kernel_profiler]
      GJ --> V[Python validators via execution_run]
      TR --> V
      EO --> DB[(kernel_db_* / map_elites.json)]
      KP --> Colab[colab-cli / SSH]
    ```
  * **AI Image Prompt:** "Software architecture org chart, dark theme. Root node 'kernel_orchestrator' in center top. Children nodes as hexagons: plan_kernel, gpu_to_jax, implement_kernel, test_generator, test_runner, kernel_profiler, kernel_mutator, evolve_orchestrator. Side panel lists Rust tools: kernel_db_init, kernel_db_sample, kernel_db_insert, kernel_db_status. Bottom cloud: Google Colab GPU/TPU via colab-cli."
* **Key Bullet Points:**
  * **Config:** `[agents.*]` blocks in `config.toml`; prompts in `.agents/prompts/*.md` (onboarded from `assets/onboarding/agents/prompts/`).
  * **Orchestration:** `subagent_spawn`, `subagent_plan_execute` with **per-step `"agent"`** field (12-step GpuToJax plan in `.agents/kernel-porting/gpu_to_jax_plan.json`).
  * **Persistence:** `kernels/projects/{id}/database/map_elites.json` + `lineage.jsonl`.
  * **Skill:** `kernel-porting` — workflow contract, mutation catalog, hardware routing.
* **Speaker Notes:**
  > Walk the audience through real names they will see in config.toml — not abstract Google ADK class names alone.

---

## Slide 8: The 12-Step GpuToJax Pipeline (Implemented)

* **Slide Title:** GpuToJax: 12-Step Plan with Gates
* **Visual Layout & Diagram Description:**
  * **Layout:** Horizontal timeline with **three color bands:**
    * Blue steps 1–4: analysis (`SIMPLIFICATION_PLAN.md`, `organized_gpu.py`)
    * Green steps 5–9: conversion + validators (`converted_jax.py`, syntax/compile/shape scripts)
    * Orange steps 10–12: tests + summary (`test_correctness.py`, `artifacts/CONVERSION_SUMMARY.md`)
  * **Gate icon** between steps 2 and 3: human **`ask_user`** plan approval (HITL).
  * **Validator callouts:** Map steps 6–11 to scripts under `skills/kernel-porting/scripts/validators/`.
  * **AI Image Prompt:** "Horizontal 12-step timeline infographic, dark theme. Steps numbered in circles. Blue section Analysis, green Conversion+Validation, orange Testing. Red human silhouette gate between step 2 and 3 labeled HITL approve plan. Small Python file icons under steps 6,8,9,11 for validator scripts. Professional tech conference style."
* **Key Bullet Points:**
  * Executed via **`subagent_plan_execute`** — each step spawns **`gpu_to_jax`** agent with prior step output in context.
  * **Artifacts per project:** documented in `kernel-porting` skill (not ad-hoc filenames).
  * **Fix loop:** `implement_kernel` agent on validator failures (MaxKernel FixConversion pattern).
  * **First demo target:** `workspace/benchmarks/vector_add/` → project `vector_add_v1`.
* **Speaker Notes:**
  > Emphasize **gates**: syntax, compile, shape, correctness — each is a script returning JSON the agent must parse.

---

## Slide 9: Validators and interpret=True

* **Slide Title:** Grounding Agents: Validators Before Silicon
* **Visual Layout & Diagram Description:**
  * **Layout:** Pipeline diagram: `converted_jax.py` → four validator boxes → green check → optional hardware path.
    1. `jax_syntax_check.py` — AST
    2. `compile_check.py` — import/`build_kernel()`
    3. `shape_check.py` — `validate_shapes()`
    4. `correctness_check.py` — pytest, `JAX_PLATFORMS=cpu`
  * **interpret=True inset:** CPU grid emulating blocks; OOB cells glow **NaN red**; caption "fail fast before TPU DMA".
  * **AI Image Prompt:** "Flowchart left to right: Python file through four validator boxes with script names, then green checkmark gate, then fork to CPU interpret grid (red NaN cells) OR TPU chip path. Dark blueprint, green success path, red failure path."
* **Key Bullet Points:**
  * Validators invoked through **`execution_run`** (local UV-managed env with JAX in `uv_requirements`).
  * **`interpret=True`** in tests catches indexing errors invisible to syntax-only checks.
  * **`jax.custom_vjp`** required in prompts for production backward passes — not optional `jax.grad`.
  * Windows dev: local path is CPU interpret; GPU/TPU profiling via **Colab or SSH** (documented in user guide).
* **Speaker Notes:**
  > This is the credibility slide — show the actual script paths in the repo.

---

## Slide 10: MAP-Elites and the Evolution Loop

* **Slide Title:** AlphaEvolve-Inspired Search: MAP-Elites in the Workspace
* **Visual Layout & Diagram Description:**
  * **Layout:** Central flywheel + 3D grid sketch (or 2D projection):
    * Axes: **latency_ms** (fitness), **complexity_loc**, **tile_volume** (see `map_elites.schema.json`).
    * Cells store elite kernels; `kernel_db_sample` reads top entries; `kernel_db_insert` writes after profile.
  * **Mutation operator wheel:** tiling | pipelining | layout | sparsity (`mutation_spec.json`).
  * **AI Image Prompt:** "Evolutionary algorithm poster. Center circular arrows: Sample, Mutate, Validate, Profile, Insert. Background heatmap grid MAP-Elites with glowing elite cells. Four mutation icons around the wheel: tile blocks, pipeline stages, layout matrix, sparse dots. Neon green on black."
* **Key Bullet Points:**
  * **Rust tools:** `kernel_db_init`, `kernel_db_sample`, `kernel_db_insert`, `kernel_db_status` (`src/tools/kernel_porting.rs`).
  * **Batch runner:** `scripts/evolve/evolve_runner.py` via **`execution_run_background`**; queue in `candidates/queue.jsonl`.
  * **Agents:** `kernel_mutator` (high temp 0.3) proposes edits; `evolve_orchestrator` coordinates loop.
  * **Literature goal:** replicate AlphaEvolve-style gains on **your** kernels — not bundled in repo.
* **Speaker Notes:**
  > Be explicit: evolution is **implemented**; headline Google numbers are **targets**, not shipped benchmarks in this repository yet.

---

## Slide 11: Hardware-in-the-Loop Profiling

* **Slide Title:** Profiling on Real Silicon: Colab, SSH, and MFU
* **Visual Layout & Diagram Description:**
  * **Layout:** Three-column routing diagram:
    | Environment | Mechanism | When |
    |-------------|-----------|------|
    | Local CPU | `execution_run` | interpret + pytest only |
    | Google Colab | `colab-cli` + `exec` | GPU/TPU latency/MFU |
    | Remote server | `[harness.execution.ssh]` | persistent lab machines |
  * **Profile script contract:** must print `RESULT_LATENCY_MS=` and optional `RESULT_TFLOPS=` for `roofline_mfu.py`.
  * **AI Image Prompt:** "Three-column architecture: laptop labeled isanagent workspace, middle arrows, right side Google Colab cloud with TPU/GPU icons and SSH server rack. Script icon profile_script.py with stdout lines RESULT_LATENCY_MS. Warning badge: always colab stop. Clean tech diagram."
* **Key Bullet Points:**
  * **`kernel_profiler` agent** writes `profile_script.py`, runs background jobs, feeds **`kernel_db_insert`**.
  * **Colab TPU path** documented in updated `colab-cli` skill (v6e/v5e install snippet).
  * **`wake_on_job_terminal`:** parent agent resumes when evolution job finishes.
  * **MFU:** computed in `roofline_mfu.py` from measured TFLOPS vs configurable peak.
* **Speaker Notes:**
  > Demo tip: live terminal showing `execution_job_status` or Colab session lifecycle if time permits.

---

## Slide 12: End-to-End Workflow (Live Demo Script)

* **Slide Title:** Demo: vector_add → Pallas → MAP-Elites
* **Visual Layout & Diagram Description:**
  * **Layout:** Single **swimlane diagram** with five lanes: User, Coordinator, kernel_orchestrator, Validators, Database.
  * **Steps numbered 1–7** matching operator guide:
    1. Load skill → spawn orchestrator
    2. `kernel_db_init(project_id=vector_add_v1, source_relative_path=benchmarks/vector_add/source/vector_add_triton.py)`
    3. `subagent_plan_execute` with gpu_to_jax plan
    4. `ask_user` approves plan
    5. correctness gate passes
    6. optional `evolve_orchestrator` + Colab profile
    7. `REPORT.md` from skill template
  * **AI Image Prompt:** "Swimlane flowchart, five horizontal lanes, seven numbered steps, dark presentation style. Highlight step 3 as 12-step sub-pipeline inset. Final artifact icons: converted_jax.py, test_correctness.py, map_elites.json, REPORT.md."
* **Key Bullet Points:**
  * All paths **sandbox-relative** under `kernels/projects/vector_add_v1/`.
  * Reference example output shape: `benchmarks/vector_add/example/converted_jax_reference.py`.
  * Benchmark ladder shipped: vector_add → matmul_relu → flash_attention → block_sparse_attention (READMEs only for advanced targets).
* **Speaker Notes:**
  > This slide is your **live demo cheat sheet** — keep terminal paths visible.

---

## Slide 13: Case Study — Block-Sparse Attention (Target Workload)

* **Slide Title:** Target Workload: Block-Sparse Attention and PrefetchScalarGridSpec
* **Visual Layout & Diagram Description:**
  * **Layout:** Memory routing diagram (retained from original deck, refined):
    * Sparse mask → Block-COO coordinates → SMEM → DMA async fetch while MXU computes previous tile.
  * **Agent connection:** `kernel_mutator` **sparsity** mutation class + `plan_kernel` reads `kernels/reference/Triton To Pallas Conversion.md`.
  * **AI Image Prompt:** "TPU memory diagram: sparse attention mask compresses to coordinate list in Scalar Memory SMEM. DMA forklift fetches Q/K blocks from HBM to SRAM while MXU works on prior tile. Label PrefetchScalarGridSpec. Orange async arrows, green compute box."
* **Key Bullet Points:**
  * **Why it matters:** irregular sparsity breaks lexicographical BlockSpec traversal — key hard case for agents.
  * **MaxEvolve mechanism:** mutation class `sparsity` in `mutation_spec.json`; not a one-shot translation.
  * **Status in repo:** benchmark README at `benchmarks/block_sparse_attention/` — **roadmap target**, not a completed port in tree.
* **Speaker Notes:**
  > Frame as **where the system should shine**, honest that the repo documents the ladder rather than claiming a finished sparse attention kernel.

---

## Slide 14: Literature Results vs. Our Engineering Baseline

* **Slide Title:** What the Literature Reports — and What We Measure Locally
* **Visual Layout & Diagram Description:**
  * **Layout:** Two-panel comparison (avoid implying we reproduced Google internal runs):
    * **Left panel — Literature (AlphaEvolve / Google):** Deepseek MLA −8.7% latency; Gemini matmul +23%; cite sources.
    * **Right panel — MaxEvolve repo today:** vector_add correctness pass; MAP-Elites insert cycle; Colab MFU readout — placeholders for **your** measured bars after running the pipeline.
  * **AI Image Prompt:** "Split slide. Left: citation-style panel with gray academic bars labeled Literature benchmarks Deepseek MLA Gemini. Right: empty bar chart template labeled Your workspace results with placeholder bars Vector add Matmul relu. Disclaimer banner: results require running MaxEvolve on your hardware."
* **Key Bullet Points:**
  * **External claims** motivate evolutionary search; **internal validation** starts at vector_add E2E.
  * Record results in `kernels/projects/{id}/REPORT.md` using skill template.
  * **`database/lineage.jsonl`** supports talk narrative of mutation genealogy.
* **Speaker Notes:**
  > Intellectual honesty builds trust — especially in an infra audience.

---

## Slide 15: Failure Modes and Recovery Loops

* **Slide Title:** When Agents Fail: Tracebacks, NaNs, and Fixes
* **Visual Layout & Diagram Description:**
  * **Layout:** **Feedback loop diagram:** Validator fail → traceback in sub-agent context → `implement_kernel` or gpu_to_jax retry → re-run gate.
  * **Side story panel:** RPAv3-style DMA OOB (literature / MaxKernel anecdote) — interpret catches analogous bugs locally as NaN patterns.
  * **AI Image Prompt:** "Circular debugging loop diagram: Fail validator, LLM reads JSON stderr, patch file, re-run, Success. Side panel red crash log DMA OOB with arrow to green interpret=True grid showing NaN at boundary. Dark SOC-style UI."
* **Key Bullet Points:**
  * **Doom loop protection:** isanagent `doom_loop_enabled` catches repeated identical tool calls.
  * **Scoped sandbox:** `resolve_path` bounds all writes to project dir.
  * **HITL:** plan approval prevents bad BlockSpec strategies early.
  * **Lesson:** closed-loop beats bigger models alone.
* **Speaker Notes:**
  > Connect to audience scars — everyone has shipped a slow or wrong kernel once.

---

## Slide 16: Conclusion and Call to Action

* **Slide Title:** Autonomous Kernel Porting as Infrastructure
* **Visual Layout & Diagram Description:**
  * **Layout:** Future roadmap arrow from **Today** (MaxEvolve in isanagent) → **Next** (FlashAttention + block-sparse benchmarks on Colab TPU) → **Future** (CI-gated kernel evolution on every PR touching hot paths).
  * **QR / link panel:** `docs/kernel-porting-user-guide.md`, `kernels/reference/`, GitHub repo.
  * **AI Image Prompt:** "Inspiring closing slide: timeline arrow left to right Today Next Future. Human engineer and holographic agent co-designing a glowing chip. Soft purple and blue lighting, minimal text, space for repo URL."
* **Key Bullet Points:**
  * **Paradigm shift:** porting = search under constraints, not transcription.
  * **Shipped today:** agents, skill, validators, MAP-Elites tools, Colab path, benchmark ladder.
  * **Try it:** `isanagent onboard` → enable `[harness.kernel_porting]` → port `benchmarks/vector_add`.
  * **Questions.**
* **Speaker Notes:**
  > Close with a single actionable invite: clone, onboard, run vector_add through the orchestrator before your next custom kernel meeting.

---

## Appendix A: Suggested Diagram-Only Slides (Optional Backup)

Use if Q&A runs long or audience wants extra depth.

1. **isanagent actor bus** — `Inbound → AgentLogic → subagent_spawn → subagent-* chat → wake_on_completion synthetic inbound`.
2. **Compiler lowering stacks** — full Triton IR vs Pallas/XLA/Mosaic (from conversion guide).
3. **MAP-Elites cell key** — `(latency bucket, complexity_loc, tile_volume)` as implemented in `kernel_db_insert`.
4. **Per-agent temperature table** — test_runner 0.0, kernel_mutator 0.3, etc. (wired at spawn via `ProviderCredentials`).

---

## Appendix B: Asset Checklist for AI Image Generation

| Slide | Primary visual | Style keywords |
|-------|----------------|----------------|
| 1 | Three-panel hero | dark, cinematic, DNA helix center |
| 2 | Decision tree | compiler OK vs kernel required |
| 3 | Roofline + conveyor | STALL metaphor, ridge 240 FLOPs/byte |
| 4 | Dual compiler stacks | orange pointer warning, green BlockSpec |
| 5 | Broken gears + wrenches | five failure modes |
| 6 | Maturity staircase | Copilot → Agent → MaxEvolve |
| 7 | Org chart | real agent names + kernel_db tools |
| 8 | 12-step timeline | HITL gate at step 2 |
| 9 | Validator pipeline | interpret NaN grid |
| 10 | MAP-Elites flywheel | mutation wheel |
| 11 | Colab / SSH routing | colab stop warning |
| 12 | Swimlane demo | vector_add_v1 paths |
| 13 | PrefetchScalarGridSpec | TPU DMA async |
| 14 | Literature vs local bars | disclaimer banner |
| 15 | Debug loop | traceback → fix |
| 16 | Roadmap arrow | repo CTA |

---

## Appendix C: Key Repo Paths (speaker quick reference)

| What | Path |
|------|------|
| Operator guide | `docs/kernel-porting-user-guide.md` |
| Skill | `workspace/skills/kernel-porting/SKILL.md` |
| 12-step plan JSON | `workspace/.agents/kernel-porting/gpu_to_jax_plan.json` |
| Validators | `workspace/skills/kernel-porting/scripts/validators/` |
| Evolution | `workspace/skills/kernel-porting/scripts/evolve/` |
| Rust DB tools | `src/tools/kernel_porting.rs` |
| Agent prompts | `workspace/.agents/prompts/` |
| Demo input | `workspace/benchmarks/vector_add/source/vector_add_triton.py` |
| Projects | `kernels/projects/{project_id}/` |
| Reference docs | `workspace/kernels/reference/` |
