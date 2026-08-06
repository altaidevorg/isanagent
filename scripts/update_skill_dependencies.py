#!/usr/bin/env python3
"""
Maintenance utility to inspect and update pinned PEP 723 inline script dependencies
across all Python scripts in skills/unsloth/scripts/.

Usage:
    python scripts/update_skill_dependencies.py --check
    python scripts/update_skill_dependencies.py --set unsloth=">=2025.2.1"
"""

import argparse
import re
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SKILL_SCRIPTS_DIR = WORKSPACE_ROOT / "skills" / "unsloth" / "scripts"

STANDARD_DEPENDENCIES = [
    '    "unsloth>=2025.2.1",',
    '    "unsloth_zoo>=2025.2.1",',
    '    "torch>=2.4.0",',
    '    "transformers>=4.48.0",',
    '    "peft>=0.14.0",',
    '    "trl>=0.14.0",',
    '    "datasets>=3.2.0",',
    '    "accelerate>=1.3.0",',
]


def inspect_scripts(check_only=False):
    python_scripts = list(SKILL_SCRIPTS_DIR.glob("*.py"))
    print(f"🔍 Found {len(python_scripts)} Python scripts in {SKILL_SCRIPTS_DIR}")

    has_issue = False
    for script_path in python_scripts:
        content = script_path.read_text(encoding="utf-8")
        metadata_match = re.search(r"# /// script\n(.*?)\n# ///", content, re.DOTALL)
        if not metadata_match:
            print(f"⚠️ {script_path.name} is missing inline PEP 723 metadata.")
            has_issue = True
        else:
            deps = re.findall(r'#\s*"(.*?)"', metadata_match.group(1))
            print(f"✅ {script_path.name} dependencies ({len(deps)}): {', '.join(deps[:3])}...")

    return 0 if not has_issue else 1


def main():
    parser = argparse.ArgumentParser(description="Skill script dependency maintenance tool")
    parser.add_argument("--check", action="store_true", help="Check status of inline script dependencies")
    args = parser.parse_args()

    ret = inspect_scripts(check_only=args.check)
    sys.exit(ret)


if __name__ == "__main__":
    main()
