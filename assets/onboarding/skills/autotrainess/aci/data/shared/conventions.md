# Shared Conventions

## Goal
Produce training data that supports the target problems or behaviors and matches the benchmark's evaluation interface as closely as possible.

## Shared rules
- Do not expand scope just to gather more data; keep work focused on the target problems or behaviors.
- Do not treat benchmark alignment as only a formatting issue; align task interface, sample structure, input-output behavior, and target capability.
- Do not use or derive training data from benchmark test answers or obviously contaminated benchmark-specific content.
- Keep each selected or constructed dataset traceable: source, transformation type, target problem, and known limitations should be explainable.
- Prefer simple, direct transformations over elaborate data pipelines unless complexity is required by the task.
- Do not use remote APIs or externally hosted model services for data construction; only local models are allowed for distillation.

## Source categories
1. Local existing data.
2. Data collected from external sources.
3. Synthetic data.
4. Model-distilled data.

## Stage boundary
- Selection identifies target data needs, initial source directions, and important construction constraints.
- Construction turns those needs and directions into a benchmark-aligned dataset and dataset description.
- Validation checks whether the constructed dataset is structurally valid, benchmark-aligned, content-valid, and ready for training.

## Return conditions
- Return to construction when the data direction is viable but the produced samples have format, schema, cleaning, synthesis, filtering, or quality issues.
- Return to selection when the target data needs or initial source directions are not viable.
