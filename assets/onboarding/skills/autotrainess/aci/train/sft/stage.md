# SFT Stage

## Purpose
Run the minimum valid supervised fine-tuning workflow for the current stage with LlamaFactory.

## Inputs
- The training dataset prepared by the data workflow.
- The benchmark-facing sample format or schema.
- A valid base model path or model id.
- Any current-stage training settings required by the workspace.

## Rules
- Use `scripts/run_llamafactory.sh`.
- Prefer full-parameter fine-tuning when the task requires it; only use PEFT when evidence and stage rules justify it.
- Keep the training data aligned with the benchmark-facing task interface.
- Start with a small validation run before scaling up.
- Do not treat training loss alone as evidence of success.
- Keep the workflow minimal for the current stage.

## Procedure
1. Review the prepared training data and its benchmark-facing format.
2. Read `shared/llamafactory.md` and confirm that LlamaFactory is usable.
3. Prepare the minimum SFT dataset assets and verify the LlamaFactory config using `shared/llamafactory.md`.
4. Run a small validation training with `scripts/run_llamafactory.sh`.
5. If the validation run is usable, continue the intended SFT run.
6. Export `final_model/` and leave it ready for evaluation.

## Decision standard
The stage is complete only when the SFT run is reproducible, the exported model is evaluation-ready, and the result is not justified by training loss alone.
