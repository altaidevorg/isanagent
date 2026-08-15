# Data Validation

## Purpose
Validate the constructed dataset and dataset description before training, then decide whether the data is ready, needs reconstruction, or requires a new selection decision.

## Required outputs
- A validation result: approve for training, return to construction, or return to selection.
- A clear description of any detected problems and the reason for the return target.

## Validation dimensions

### 1. Structural validity
Check whether the data is mechanically usable:
- schema or template correctness
- field completeness and consistency
- readable encoding
- no broken, empty, truncated, or obviously malformed samples

### 2. Benchmark alignment
Check whether the data is training the right thing:
- input form matches the benchmark-facing task
- output form matches the benchmark-facing task
- constructed samples match the rendered evaluation-style examples
- answer boundaries and final-answer location match the rendered evaluation-style examples
- sample structure is compatible with the evaluation interface
- target behavior supports the target problems or behaviors described for the dataset

### 3. Content quality
Check whether the data itself is trustworthy:
- no obvious garbage text or unreadable content
- no clear sample errors
- no obvious low-value duplication
- no synthetic samples that are overly templated or unrealistic
- no leakage or contamination risk

## Rules
- Do not approve data only because the format looks correct; verify that the task objective is aligned as well.
- Do not approve data if sampled training examples do not match the rendered evaluation-style examples in input style, output style, answer boundary, and final-answer location.
- Separate construction errors from selection-direction errors.
- Return to construction if the direction is viable but the produced samples are flawed.
- Return to selection if the target data needs or initial source directions are not viable.

## Procedure
1. Inspect the constructed dataset and dataset description.
2. Check structural correctness, including schema, required fields, encoding, and malformed samples.
3. Compare several constructed training samples against the rendered evaluation-style examples.
4. Check whether the dataset matches the benchmark evaluation interface and target behaviors.
5. Review sample quality and look for garbage, corruption, duplication, leakage risk, or unrealistic synthesis.
6. Decide whether any detected problem belongs to construction or selection.
7. Produce one of three decisions:
 - approve for training
 - return to construction
 - return to selection

## Decision standard
The stage is complete when the dataset is approved for training or sent back with a clear reason and return target.
