---
name: autotrainess
description: Autonomous LLM post-training via AutoTrainHub-style ACI — plan, data, LlamaFactory train, eval, and experiment logging with named sub-agents.
requires:
  bins: ["uv"]
  env: []
always: false
---

# AutoTrainess Post-Training

Native isanagent workflow adapted from [AutoTrainess / AutoTrainHub](https://github.com/simple-agent-lab/AutoTrainess) (MIT). Turns post-training into a closed loop:

```text
iteration_plan -> data -> train -> eval -> log
```

## Attribution

ACI instructions and LlamaFactory scripts are adapted from simple-agent-lab/AutoTrainess (MIT License). Keep the spirit of the upstream hard constraints when running competitive or benchmark-oriented post-training.

## When to load

User asks for autonomous fine-tuning, post-training, SFT/RL iteration on a benchmark, or AutoTrainess-style training loops. Delegate to `train_orchestrator` via `subagent_spawn(agent="train_orchestrator", ...)`.

## Hard rules (never violate)

1. **No benchmark leakage** — do not use benchmark examples, answers, or contaminated overlap for training data.
2. **Base model only** — fine-tune only from the task base model or checkpoints derived from it (not instruct/chat/larger/different models as start or merge source).
3. **No generation-config gaming** — never tune generation config solely to inflate benchmark scores.
4. **LlamaFactory only** — stay inside LlamaFactory for training; do not switch frameworks or write custom training loops.
5. **Evidence-based iterations** — each Stage 2/3 iteration needs a hypothesis, intervention, and success criterion from prior eval evidence.
6. **Respect GPU assignment** — if `CUDA_VISIBLE_DEVICES` is set, do not override it.

## Project layout

```
train/projects/{project_id}/
  database/meta.json
  database/iterations.jsonl
  experiment_log.md
  iterations/{iteration_id}/
    iteration_plan.md
    data/
    train/
    eval/
  final_model/                 # current evaluation-ready export
  eval_results/                # latest eval artifacts (also per-iteration under iterations/)
  artifacts/
  REPORT.md
```

Initialize with `train_db_init(project_id=..., base_model=..., benchmark=..., target_metric=...)`.

## Stages

### Stage 0: Task definition
Define target, evaluation entrypoint, and resource constraints. Call `train_db_init`.

### Stage 1: Base model evaluation
Run the real benchmark on the base model. Log with `train_db_append`. Stop only if an explicit target is already met.

### Stage 2: Local diagnosis and optimization
Full iterations with a clear local hypothesis. Prefer simple changes.

### Stage 3: Evidence-guided exploration
Expand search (datasets, methods) guided by Stage 2 evidence. Still run full local iterations.

## Full iteration (Stage 2/3)

Use `subagent_plan_execute` with `.agents/autotrainess/iteration_plan.json` (or `skills/autotrainess/iteration_plan.json`):

1. `train_planner` — write `iterations/{id}/iteration_plan.md` (see `aci/iteration_plan.md`)
2. `data_prep` — selection → construction → validation (`aci/data/`)
3. `trainer` — LlamaFactory SFT/RL (`aci/train/`); export project `final_model/`
4. `train_evaluator` — real benchmark eval (`aci/eval/`)

Then the orchestrator calls `train_db_append` with concrete metrics and next action.

## ACI docs (read during the matching stage)

| Stage | Doc |
|-------|-----|
| Plan | `aci/iteration_plan.md` |
| Data | `aci/data/SKILL.md` + selection/construction/validation |
| Train | `aci/train/SKILL.md` + `aci/train/shared/llamafactory.md` |
| Eval | `aci/eval/SKILL.md` |
| Log | `aci/log.md` + `train_db_append` |

## Training compute

- Prefer `[harness.execution.ssh]` + `execution_run_background` for GPU training.
- Local CPU is for smoke / config validation only.
- Install/run helpers: `aci/train/scripts/install_llamafactory.sh`, `run_llamafactory.sh`.

## Behavior taxonomy (guidance)

See paper Appendix B labels when analyzing agent choices: evaluation (E1–E3), input format (T1–T3), output format (O1–O3), data (D1–D6), training (U1–U7), planning (P1–P4). Use as vocabulary in plans/logs; not enforced by tools.

## Delivery

Copy best checkpoint path into ledger, write `REPORT.md` from `templates/REPORT.md`.
