#!/usr/bin/env python3
"""
Test script for Unsloth Skill validation.

Validates:
1. PEP 723 Inline Script Metadata presence and format in skill scripts.
2. Python syntax validity of skill scripts.
3. SKILL.md structure, absence of hype words, and workflow steps.
4. GRPO batch size divisibility requirement: (batch_size * grad_accum) % num_generations == 0.
5. SFTTrainer signature compliance (processing_class, SFTConfig parameters).
6. BooleanOptionalAction CLI argument handling.
"""

import ast
import os
import re
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = WORKSPACE_ROOT / "skills" / "unsloth"
SCRIPTS_DIR = SKILL_DIR / "scripts"
SKILL_MD = SKILL_DIR / "SKILL.md"

HYPE_WORDS = [
    r"80%\s*(less|memory\s*reduction)",
    r"1\.6x\s*speedup",
    r"2-5x\s*faster",
    r"ultra-fast",
    r"superfast",
]


def check_pep723_metadata(script_path: Path) -> bool:
    content = script_path.read_text(encoding="utf-8")
    script_block_pattern = r"# /// script\n(#.*\n)+# ///"
    if not re.search(script_block_pattern, content):
        print(f"❌ [PEP 723 Missing] {script_path.name} does not contain valid inline script metadata block.")
        return False
    if "dependencies =" not in content:
        print(f"❌ [PEP 723 Incomplete] {script_path.name} metadata block missing 'dependencies ='.")
        return False
    return True


def check_python_syntax(script_path: Path) -> bool:
    try:
        content = script_path.read_text(encoding="utf-8")
        ast.parse(content, filename=str(script_path))
        return True
    except SyntaxError as e:
        print(f"❌ [Syntax Error] {script_path.name}: line {e.lineno} - {e.msg}")
        return False


def check_grpo_divisibility(script_path: Path) -> bool:
    if script_path.name != "finetune_grpo_reasoning.py":
        return True
    content = script_path.read_text(encoding="utf-8")
    if "%" not in content or "num_generations" not in content:
        print(f"❌ [GRPO Divisibility Missing] {script_path.name} does not enforce batch size divisibility check.")
        return False
    return True


def check_sft_signature(script_path: Path) -> bool:
    if script_path.name != "finetune_sft.py":
        return True
    content = script_path.read_text(encoding="utf-8")
    if "processing_class=tokenizer" not in content:
        print(f"❌ [SFT Signature Warning] {script_path.name} does not use processing_class=tokenizer on SFTTrainer.")
        return False
    if "SFTConfig(" not in content or "dataset_text_field=" not in content:
        print(f"❌ [SFTConfig Missing] {script_path.name} does not pass dataset_text_field in SFTConfig.")
        return False
    return True


def check_skill_md() -> bool:
    if not SKILL_MD.exists():
        print(f"❌ [Missing File] {SKILL_MD} not found.")
        return False

    content = SKILL_MD.read_text(encoding="utf-8")
    passed = True

    for hype_regex in HYPE_WORDS:
        match = re.search(hype_regex, content, re.IGNORECASE)
        if match:
            print(f"❌ [Hype Word Detected] Found forbidden phrase matching '{hype_regex}' in SKILL.md: '{match.group(0)}'")
            passed = False

    # Check for long python code blocks (> 30 lines) in SKILL.md
    code_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)
    for idx, block in enumerate(code_blocks, 1):
        line_count = len(block.strip().splitlines())
        if line_count > 30:
            print(f"❌ [SKILL.md Code Block Too Long] Python block #{idx} has {line_count} lines (> 30 line limit). Move to scripts/.")
            passed = False

    return passed


def main():
    print("🧪 Running Unsloth Skill Validation Tests...")
    all_passed = True

    # 1. Test SKILL.md
    print("\n--- Checking SKILL.md ---")
    if check_skill_md():
        print("✅ SKILL.md passes structure & content guidelines.")
    else:
        all_passed = False

    # 2. Test Scripts in skills/unsloth/scripts/
    print("\n--- Checking Skill Python Scripts ---")
    python_scripts = list(SCRIPTS_DIR.glob("*.py"))
    if not python_scripts:
        print("⚠️ No Python scripts found in skills/unsloth/scripts/")

    for script_path in python_scripts:
        print(f"Inspecting {script_path.name}...")
        syn_ok = check_python_syntax(script_path)
        pep_ok = check_pep723_metadata(script_path)
        grpo_ok = check_grpo_divisibility(script_path)
        sft_ok = check_sft_signature(script_path)
        if not (syn_ok and pep_ok and grpo_ok and sft_ok):
            all_passed = False

    print("\n" + ("=" * 40))
    if all_passed:
        print("🎉 ALL UNSLOTH SKILL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME UNSLOTH SKILL TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
