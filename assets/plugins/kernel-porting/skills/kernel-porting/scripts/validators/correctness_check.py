# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax>=0.4.30",
#   "jaxlib>=0.4.30",
#   "pytest>=8.0",
# ]
# ///
"""Run pytest correctness suite with interpret=True (MaxEvolve step 11)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_path",
        type=Path,
        help="Path to test_correctness.py or test directory",
    )
    args = parser.parse_args()
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env.setdefault("PYTEST_ADDOPTS", "-q")
    proc = subprocess.run(
        ["uv", "run", "pytest", str(args.test_path), "-q"],
        capture_output=True,
        text=True,
        env=env,
    )
    ok = proc.returncode == 0
    print(
        json.dumps(
            {
                "ok": ok,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-4000:],
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
