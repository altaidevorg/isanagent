# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Estimate latency, TFLOPS, and MFU from a profiling script output or micro-benchmark."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile_script",
        type=Path,
        help="PEP 723 Python script that prints RESULT_LATENCY_MS= and optional RESULT_TFLOPS=",
    )
    parser.add_argument(
        "--peak_tflops",
        type=float,
        default=100.0,
        help="Hardware peak TFLOPS for MFU",
    )
    args = parser.parse_args()
    start = time.perf_counter()
    proc = subprocess.run(
        ["uv", "run", str(args.profile_script)],
        capture_output=True,
        text=True,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    latency_ms = wall_ms
    tflops = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_LATENCY_MS="):
            latency_ms = float(line.split("=", 1)[1].strip())
        if line.startswith("RESULT_TFLOPS="):
            tflops = float(line.split("=", 1)[1].strip())
    mfu = None
    if tflops is not None and args.peak_tflops > 0:
        mfu = tflops / args.peak_tflops
    print(
        json.dumps(
            {
                "ok": proc.returncode == 0,
                "latency_ms": latency_ms,
                "tflops": tflops,
                "mfu": mfu,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-1000:],
            }
        )
    )
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
