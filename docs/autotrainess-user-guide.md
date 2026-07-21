# AutoTrainess Post-Training — Operator Guide

AutoTrainess ports the AutoTrainHub agent-computer interface into isanagent: a closed loop of **plan → data → train → eval → log** with named sub-agents and an experiment ledger. ACI text and LlamaFactory scripts are adapted from [simple-agent-lab/AutoTrainess](https://github.com/simple-agent-lab/AutoTrainess) (MIT).

## Enable

In `config.toml`:

```toml
[harness.autotrainess]
enabled = true
default_project_root = "train/projects"
max_log_entries = 500

[harness.subagents]
enabled = true

[harness.execution]
enabled = true
# Prefer SSH GPU for real training:
# default_provider = "ssh"
# [harness.execution.ssh]
# ...
```

Onboarding copies the `autotrainess` skill, agent prompts, and `.agents/autotrainess/iteration_plan.json`.

## Tools

| Tool | Purpose |
|------|---------|
| `train_db_init` | Create `train/projects/{id}/` layout + empty ledger |
| `train_db_append` | Append one iteration to JSONL + `experiment_log.md` |
| `train_db_list` | List recent iterations |
| `train_db_status` | Stage, best metric, paths |
| `train_db_get` | Fetch one iteration by id |

## Named agents

| Agent | Role |
|-------|------|
| `train_orchestrator` | Stages 0–3, plan execute, ledger |
| `train_planner` | Evidence-based `iteration_plan.md` |
| `data_prep` | Selection → construction → validation |
| `trainer` | LlamaFactory SFT/RL → `final_model/` |
| `train_evaluator` | Real benchmark eval + failure modes |

## Typical session

1. User states base model, benchmark/eval entrypoint, and success criterion.
2. Coordinator loads `autotrainess` → spawns `train_orchestrator`.
3. Orchestrator runs `train_db_init`, then Stage 1 baseline eval.
4. For each Stage 2/3 iteration: `subagent_plan_execute` with `.agents/autotrainess/iteration_plan.json` (replace `{PROJECT_ID}` / `{ITERATION_ID}`), then `train_db_append`.
5. Deliver `REPORT.md` from `skills/autotrainess/templates/REPORT.md`.

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
  final_model/
  eval_results/
  artifacts/
  REPORT.md
```

## Training backend

- Fixed to **LlamaFactory** (`hiyouga/LlamaFactory`).
- Scripts: `skills/autotrainess/aci/train/scripts/install_llamafactory.sh`, `run_llamafactory.sh`.
- Prefer `execution_run_background` on an SSH provider for GPU jobs.
- Local CPU is for smoke / config validation only.

## Hard constraints (from AutoTrainess)

- No benchmark leakage into training data.
- Fine-tune only from the task base model or its derived checkpoints.
- Do not tune generation config to inflate scores.
- Stay inside LlamaFactory (no framework hopping).

## ACI docs

Under `skills/autotrainess/aci/`: `iteration_plan.md`, `data/`, `train/`, `eval/`, `log.md`.

## Windows note

Build isanagent in **release** mode on Windows. Real training should target a remote Linux GPU via SSH execution.

## See also

- [Execution user guide](execution-user-guide.md)
- [Kernel porting user guide](kernel-porting-user-guide.md) (same domain-workflow pack pattern)
- Upstream: https://github.com/simple-agent-lab/AutoTrainess
