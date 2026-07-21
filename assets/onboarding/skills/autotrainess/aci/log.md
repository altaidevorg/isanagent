# log

## Task
After each completed iteration, append one new entry via `train_db_append` (preferred) so both `database/iterations.jsonl` and `experiment_log.md` stay in sync. If tools are unavailable, append manually to `experiment_log.md` using the format below.

## Rules
- If `experiment_log.md` does not exist, create it. Prefer `train_db_init` first.
- Each call should record only the iteration that has just finished.
- Organize the log by stage.

## Entry Format

### Iteration: <id>

- Context: <stage, objective, or current focus>
- Status: completed | failed | blocked
- Motivation: <why this iteration>
- References: <papers, docs, repos, datasets, blogs, or notes; "None" if unused>
- Starting checkpoint: <path or model id>
- Training data: <path and size/notes>
- Method: <SFT / RL / etc.>
- Training config: <path or key hyperparameters>
- Evaluation: <protocol and sample count>
- Result: <metrics>
- Analysis: <what the result means>
- Artifacts: <paths>
- Next action: <concrete follow-up>

## Field rules
- Fill every field. Use `None` or `N/A` only when the field truly does not apply.
- Record concrete evidence: exact metrics, commands, paths, and dataset sizes.
- If the iteration failed or was blocked, record the specific cause.
- The next action must follow from the recorded result and analysis.
