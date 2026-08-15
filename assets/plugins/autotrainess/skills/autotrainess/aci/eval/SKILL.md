# eval

## Purpose
Run the benchmark's real evaluation on `final_model/` and record reproducible evidence needed for the next stage decision.

## Inputs
- Project under `train/projects/{project_id}/`.
- `final_model/` at the project root.

## Required outputs
- Per-iteration `iterations/{iteration_id}/eval/` and/or project `eval_results/` with raw outputs or logs.
- The exact evaluation command or config used.
- A concise metrics summary.
- `sample_summary.md` with 15 randomly selected evaluation samples, including score, input, target, and model output.
- A brief note on the main 1–3 observed failure modes and whether each looks more like a data problem, a training problem, or an inference/template problem.

## Rules
- Use the benchmark's real evaluation entrypoint.
- If evaluation fails, stay in the benchmark's real evaluation workflow, debug the failure, and retry.
- For any evaluation used to compare checkpoints, judge model quality, or choose the next iteration, use at least `max(32, ceil(5% of the benchmark))` samples. If the benchmark has fewer than 32 samples, evaluate the full benchmark.
- Runs below that sample floor are allowed only as smoke tests; do not use them as evidence that one checkpoint or approach is better.
- Always produce `sample_summary.md` with 15 random evaluation samples.
- Use `aci/eval/scripts/summarize_eval_samples.py` when the benchmark outputs compatible `inspect_ai` logs; otherwise, add the minimum benchmark-specific script or logging needed.
- Keep the output focused on evidence needed for the next decision.

## Procedure
1. Locate the canonical evaluation entrypoint.
2. If using a limited evaluation, determine the benchmark sample count and choose a limit that satisfies the sample-floor rule.
3. Run evaluation on `final_model/`.
4. Save raw outputs, commands, the sample count or limit used, and a concise metrics summary.
5. If evaluation fails, debug inside the real evaluation workflow, then retry with the minimum necessary fix.
6. Generate `sample_summary.md` with 15 random samples.
7. Summarize the main 1–3 observed failure modes (data / training / inference-template).
