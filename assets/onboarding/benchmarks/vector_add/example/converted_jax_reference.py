# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax>=0.4.30",
#   "jaxlib>=0.4.30",
# ]
# ///
"""Reference Pallas-style vector add for MaxEvolve Phase 1 (interpret-friendly)."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def vector_add_reference(x, y):
    return x + y


def build_kernel():
    return jax.jit(vector_add_reference)


def validate_shapes():
    x = jnp.ones((128,), dtype=jnp.float32)
    y = jnp.ones((128,), dtype=jnp.float32)
    out = build_kernel()(x, y)
    assert out.shape == x.shape
    return True
