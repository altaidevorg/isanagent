# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax>=0.4.30",
#   "jaxlib>=0.4.30",
# ]
# ///
"""Profile script template — prints RESULT_* lines for roofline_mfu.py.

Copy to kernels/projects/{id}/profile_script.py and customize the benchmark loop.
Run: uv run profile_script.py
Or: uv run skills/kernel-porting/scripts/profile/roofline_mfu.py profile_script.py
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp


def main() -> None:
    # Replace with your jitted kernel from converted_jax.py
    fn = jax.jit(lambda x, y: x + y)
    x = jnp.ones((1024,), dtype=jnp.float32)
    y = jnp.ones((1024,), dtype=jnp.float32)
    fn(x, y).block_until_ready()

    start = time.perf_counter()
    for _ in range(100):
        fn(x, y).block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 100.0

    print(f"RESULT_LATENCY_MS={elapsed_ms:.6f}")
    # Optional: print(f"RESULT_TFLOPS={tflops:.4f}")


if __name__ == "__main__":
    main()
