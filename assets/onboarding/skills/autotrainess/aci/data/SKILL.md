# data

## Purpose
Prepare training data that addresses real problems exposed by previous training or evaluation, aligns with the benchmark evaluation interface, and is ready for downstream training.

## Core principles
- Drive all data work from concrete problems found in previous training or evaluation.
- Prioritize alignment with the benchmark's real evaluation interface over broad or generic data expansion.
- Prefer the smallest effective dataset that addresses the current problems.
- Keep data sources, transformations, and synthetic generation traceable.
- Avoid benchmark leakage and contaminated data.

## Workflow
1. Read [shared/conventions.md](./shared/conventions.md) for shared rules.
2. Run [selection/stage.md](./selection/stage.md) to identify target data needs and initial source directions.
3. Run [construction/stage.md](./construction/stage.md) to turn those needs and directions into a benchmark-aligned training dataset.
4. Run [validation/stage.md](./validation/stage.md) for data validation before training.
5. If validation finds construction issues, return to construction. If validation finds target-need or source-direction issues, return to selection.

## Required outputs
- A final training dataset under `iterations/{iteration_id}/data/` ready for LlamaFactory.
- A concise dataset description covering target problems, data sources, sample format, known limitations, and validation status.
