# Vector add benchmark (MaxEvolve Phase 1 E2E)

Use this as the first porting target:

1. `kernel_db_init(project_id="vector_add_v1", source_relative_path="benchmarks/vector_add/source/vector_add_triton.py")`
2. Run GpuToJax 12-step plan
3. Pass `test_correctness.py` with `interpret=True`
4. Optional evolution on `block_shape` / grid tiling

Reference CPU behavior can be implemented in tests without GPU.
