# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Batch evolution driver — validates candidates and records fitness proxies.

Designed to run under execution_run_background. LLM mutations are applied
offline by the kernel_mutator agent; this script processes a mutation queue file.

Run from sandbox root:
  uv run skills/kernel-porting/scripts/evolve/evolve_runner.py --project vector_add_v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from map_elites import EliteCell, insert_cell, load_archive, save_archive


def run_validator(script: Path, *args: str) -> dict:
    cmd = ["uv", "run", str(script), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"ok": False, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Project id under kernels/projects/")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("kernels/projects"),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        help="JSONL file of candidate records {id,kernel_path,mutation_class,parent_id,generation,fitness_latency_ms}",
    )
    parser.add_argument("--generations", type=int, default=1)
    args = parser.parse_args()

    root = args.project_root / args.project
    db_path = root / "database" / "map_elites.json"
    archive = load_archive(db_path)
    validators = Path(__file__).resolve().parent.parent / "validators"
    queue_path = args.queue or (root / "candidates" / "queue.jsonl")
    if not queue_path.is_file():
        print(json.dumps({"ok": False, "error": f"queue not found: {queue_path}"}))
        return 1

    processed = 0
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kernel = root / rec["kernel_path"]
        syn = run_validator(validators / "jax_syntax_check.py", str(kernel))
        if not syn.get("ok"):
            continue
        dest_kernel = root / "converted_jax.py"
        backup_kernel = root / "converted_jax.py.bak"
        has_backup = dest_kernel.is_file()
        if has_backup:
            dest_kernel.rename(backup_kernel)
        try:
            shutil.copy(kernel, dest_kernel)
            corr = run_validator(
                validators / "correctness_check.py",
                str(root / "test_correctness.py"),
            )
        finally:
            if has_backup:
                if dest_kernel.is_file():
                    dest_kernel.unlink()
                backup_kernel.rename(dest_kernel)
            elif dest_kernel.is_file():
                dest_kernel.unlink()
        if not corr.get("ok"):
            continue
        cell = EliteCell(
            id=rec["id"],
            kernel_path=rec["kernel_path"],
            fitness_latency_ms=rec.get("fitness_latency_ms"),
            fitness_mfu=rec.get("fitness_mfu"),
            fitness_tflops=rec.get("fitness_tflops"),
            complexity_loc=rec.get("complexity_loc"),
            tile_volume=rec.get("tile_volume"),
            mutation_class=rec.get("mutation_class"),
            parent_id=rec.get("parent_id"),
            generation=rec.get("generation"),
        )
        insert_cell(archive, cell)
        processed += 1
        if processed >= args.generations:
            break

    save_archive(db_path, archive)
    print(
        json.dumps(
            {
                "ok": True,
                "processed": processed,
                "global_best_id": archive.get("global_best_id"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
