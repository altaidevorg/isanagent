# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax>=0.4.30",
#   "jaxlib>=0.4.30",
# ]
# ///
"""Lightweight shape sanity checks for converted JAX kernels."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_path", type=Path)
    parser.add_argument("--entry", default="validate_shapes")
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(args.kernel_path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("kernel_module", args.kernel_path)
    if spec is None or spec.loader is None:
        print(json.dumps({"ok": False, "error": "failed to load module spec"}))
        return 1
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"import failed: {exc}"}))
        return 1
    fn = getattr(mod, args.entry, None)
    if fn is None:
        print(
            json.dumps(
                {
                    "ok": True,
                    "validated": False,
                    "note": f"define `{args.entry}()` in kernel for automated shape checks",
                }
            )
        )
        return 0
    try:
        result = fn()
        print(json.dumps({"ok": True, "validated": True, "result": str(result)}))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"shape validation failed: {exc}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
