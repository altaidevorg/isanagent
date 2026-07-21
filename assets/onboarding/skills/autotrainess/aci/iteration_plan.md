# iteration_plan

## Purpose
Define a clear goal and concrete action plan for the current experiment iteration based on real evidence from previous experiments.

## Inputs
- Results from previous experiments (`train_db_list`, `experiment_log.md`).
- Prior evaluation evidence and analysis.
- The current training and data context under `train/projects/{project_id}/`.

## Required outputs
Write `iterations/{iteration_id}/iteration_plan.md` containing:
- The main problems observed in previous experiments.
- The main objective of the current iteration.
- The changes planned for the current iteration.
- Whether this iteration mainly changes data, training, or both.
- The outcome that will count as success.
- Concise guidance for downstream data or training work.

## Rules
- Base the plan on real evidence rather than speculation.
- Focus on one main objective rather than every issue at once.
- Separate previous problems, current objective, planned changes, and success criteria clearly.
- Define direction only — do not execute data construction or training in this stage.

## Procedure
1. Review previous experiment results and identify the main problems.
2. Decide what the current iteration is mainly trying to improve.
3. Define the main changes to make in this iteration.
4. State what outcome will count as success for this iteration.
5. Provide concise guidance for downstream data and training work.
