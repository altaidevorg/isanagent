# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Validate Python/JAX syntax for a kernel file (MaxEvolve GpuToJax step 6)."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_path", type=Path)
    args = parser.parse_args()
    path = args.kernel_path
    if not path.is_file():
        print(json.dumps({"ok": False, "error": f"file not found: {path}"}))
        return 1
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "line": exc.lineno}))
        return 1
    imports = [
        n.names[0].name if hasattr(n, "names") else getattr(n, "module", "")
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(path),
                "line_count": len(source.splitlines()),
                "import_count": len(imports),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
