# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax>=0.4.30",
#   "jaxlib>=0.4.30",
# ]
# ///
"""Attempt JAX import and optional pallas_call compile (MaxEvolve step 8)."""
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
    parser.add_argument(
        "--entry",
        default="build_kernel",
        help="Function name that returns a jitted callable or pallas_call wrapper",
    )
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        import jax  # noqa: F401
    except ImportError as exc:
        print(json.dumps({"ok": False, "error": f"jax not installed: {exc}"}))
        return 1
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
                    "compiled": False,
                    "note": f"no entry function '{args.entry}'; import-only pass",
                }
            )
        )
        return 0
    try:
        fn()
        print(json.dumps({"ok": True, "compiled": True, "entry": args.entry}))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"compile failed: {exc}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
