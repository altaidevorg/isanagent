# Data Construction

## Purpose
Turn the selected data needs and initial source directions into a benchmark-aligned training dataset.

## Required outputs
- A training dataset that is ready for downstream training.
- A concise dataset description covering source origins, transformations, target problems or behaviors, sample format, known limitations, and construction decisions.

## Rules
- Build data that supports the target problems or benchmark-facing behaviors, not generic capability expansion.
- Use the benchmark's `evaluate.py`, `templates/`, `task_context/`, or similar files when present to inspect the exact model-facing input and expected output form before deciding the training sample format.
- Align the training data with the benchmark's task interface, sample structure, input-output behavior, answer boundary, and final-answer location.
- Treat source directions from selection as starting points; return to selection if they cannot support the target data needs.
- Do not use any remote API or externally hosted model service for data construction; only local models are allowed.
- When local or externally collected data differs substantially from the benchmark distribution, consider bounded, explainable synthetic or model-distilled data that stays tied to the target data needs.
- Keep the dataset large enough to cover the target problems or behaviors, but small enough to avoid unnecessary noise, redundancy, and training cost.

## Construction actions
The agent may perform the following actions as needed:
- extract usable samples from source data
- clean corrupted or noisy data
- remove exact or obvious low-value duplicates
- reduce redundant or weakly relevant samples to keep the dataset focused
- rewrite or restructure samples to match the benchmark-facing format
- synthesize new samples
- distill new samples from local model outputs
- normalize fields into a consistent training schema

## Procedure
1. Review the target problems, source directions, constraints, and risks passed from selection.
2. Inspect the benchmark evaluation path and render or reconstruct several evaluation-style examples when possible.
3. Decide the target training sample format from the observed model-facing input, expected output form, answer boundary, and final-answer location.
4. Inspect candidate sources and decide whether they can support the target data needs. If they are viable, continue construction; if not, return to selection.
5. Extract, clean, rewrite, restructure, synthesize, or distill samples as needed.
6. Filter out broken, unreadable, empty, duplicated, misaligned, or clearly low-value samples, then reduce redundant or weakly relevant samples to keep the dataset focused.
7. Produce the final dataset and dataset description.

## Decision standard
The stage is complete when the dataset is usable for training, aligned with the benchmark-facing task, and described well enough for validation.
