# Data Selection

## Purpose
Identify the data needs suggested by observed problems and choose initial source directions for construction.

## Required outputs
- The target problems or behaviors the new data should support.
- Initial source directions for construction.
- Important constraints or risks for construction, such as benchmark alignment, leakage risk, or source limitations.

## Rules
- Let observed problems guide the data direction.
- Avoid source directions that are clearly misaligned with the benchmark, low quality, or likely to introduce leakage or contamination.

## Procedure
1. Review available evidence from prior training, evaluation, or benchmark misses.
2. Identify the data needs implied by those problems or required benchmark-facing behaviors.
3. Choose initial source directions, such as local data, external data, synthetic data, or model-distilled data. If local or external data is substantially different from the benchmark distribution, consider synthetic or model-distilled data as source directions.
4. Pass unresolved assumptions, source limitations, leakage risks, and construction constraints to the construction stage.
