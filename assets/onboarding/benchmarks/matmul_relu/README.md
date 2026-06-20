# Matmul + ReLU benchmark (MaxEvolve Phase 2 evolution)

Tiled matmul with elementwise ReLU — use for MAP-Elites tile parameter search after correctness gate passes.

Evolution targets: `block_m`, `block_n`, `max_concurrent_steps` (Mosaic GPU).
