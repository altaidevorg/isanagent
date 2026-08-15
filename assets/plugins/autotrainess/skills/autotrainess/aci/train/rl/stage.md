# RL Stage

## Purpose
Run the minimum valid RL workflow for the current stage with LlamaFactory only when current evidence supports using RL.

## Inputs
- Recent evaluation evidence showing why RL is needed.
- The current model or base model to continue from.
- The minimum reward definition, feedback signal, or RL data required by the selected setup.
- Any current-stage constraints required by the workspace.

## Rules
- Use RL only when current evidence supports it.
- Use `scripts/run_llamafactory.sh`.
- State the reward or feedback signal actually used in practice.
- Start with a small validation run before scaling up.
- Stay inside the LlamaFactory workflow if failures occur.

## Procedure
1. Review the latest evaluation evidence and confirm that RL is justified.
2. Read `shared/llamafactory.md` and confirm that LlamaFactory is usable.
3. Prepare the minimum reward setup or RL data, and verify the LlamaFactory config using `shared/llamafactory.md`.
4. Run a small validation RL run with `scripts/run_llamafactory.sh`.
5. If the validation run is usable, continue the intended RL run.
6. Export `final_model/` and leave it ready for evaluation.

## Decision standard
The stage is complete only when RL is justified by current evidence, the run is reproducible, and the exported model is ready for real evaluation.
