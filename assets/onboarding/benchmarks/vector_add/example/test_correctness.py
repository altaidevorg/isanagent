# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax>=0.4.30",
#   "jaxlib>=0.4.30",
#   "pytest>=8.0",
# ]
# ///
"""Correctness tests for converted_jax_reference.py (run: uv run test_correctness.py)."""
from __future__ import annotations

import jax.numpy as jnp


def test_vector_add_reference():
    from converted_jax_reference import build_kernel

    x = jnp.linspace(0, 1, 16)
    y = jnp.linspace(1, 2, 16)
    out = build_kernel()(x, y)
    assert jnp.allclose(out, x + y)
