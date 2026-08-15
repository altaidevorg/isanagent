#!/usr/bin/env python3
"""MCP Stdio Server for MaxEvolve Kernel Porting (MAP-Elites Database Tools)."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def get_project_root(project_id: str, default_root: Optional[str] = None) -> Path:
    base = default_root or os.environ.get("ISANAGENT_PROJECT_ROOT") or "kernels/projects"
    return Path(base) / project_id.strip()


def get_db_path(project_id: str, default_root: Optional[str] = None) -> Path:
    return get_project_root(project_id, default_root) / "database" / "map_elites.json"


def read_archive(project_id: str, default_root: Optional[str] = None) -> Dict[str, Any]:
    db_file = get_db_path(project_id, default_root)
    if not db_file.exists():
        raise FileNotFoundError(
            f"MAP-Elites archive not found at {db_file}. Run kernel_db_init first."
        )
    with open(db_file, "r", encoding="utf-8") as f:
        return json.load(f)


def write_archive_atomic(path: Path, archive: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2)
    tmp.replace(path)


def update_global_best(archive: Dict[str, Any]) -> None:
    cells = archive.get("cells", {})
    best_id = None
    best_latency = float("inf")
    for cid, cell in cells.items():
        lat = cell.get("fitness_latency_ms")
        if lat is not None and lat < best_latency:
            best_latency = lat
            best_id = cid
    archive["global_best_id"] = best_id


# --- Tool Implementations ---


def kernel_db_init(
    project_id: str,
    target_hardware: str,
    default_root: Optional[str] = None,
) -> Dict[str, Any]:
    root = get_project_root(project_id, default_root)
    (root / "database").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "candidates").mkdir(parents=True, exist_ok=True)
    (root / "source").mkdir(parents=True, exist_ok=True)

    db_file = root / "database" / "map_elites.json"
    archive = {
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "target_hardware": target_hardware,
        "dimensions": [
            "fitness_latency_ms",
            "complexity_loc",
            "tile_volume",
        ],
        "bins_per_dimension": [10, 5, 5],
        "cells": {},
        "global_best_id": None,
    }
    write_archive_atomic(db_file, archive)
    return {
        "status": "initialized",
        "project_id": project_id,
        "target_hardware": target_hardware,
        "project_root": str(root),
        "db_path": str(db_file),
    }


def kernel_db_sample(
    project_id: str,
    strategy: str,
    batch_size: int = 1,
    mutation_class: Optional[str] = None,
    default_root: Optional[str] = None,
) -> Dict[str, Any]:
    archive = read_archive(project_id, default_root)
    cells = archive.get("cells", {})
    if not cells:
        return {
            "project_id": project_id,
            "strategy": strategy,
            "sampled_cells": [],
            "message": "Archive is empty. Insert seed kernels first.",
        }

    items = list(cells.values())
    if mutation_class:
        filtered = [c for c in items if c.get("mutation_class") == mutation_class]
        if filtered:
            items = filtered

    if strategy == "fastest":
        items.sort(key=lambda c: c.get("fitness_latency_ms") or float("inf"))
    elif strategy == "curiosity":
        items.sort(key=lambda c: c.get("generation") or 0)
    else:  # random / fallback
        import random

        random.shuffle(items)

    sampled = items[: max(1, batch_size)]
    return {
        "project_id": project_id,
        "strategy": strategy,
        "count": len(sampled),
        "sampled_cells": sampled,
        "global_best_id": archive.get("global_best_id"),
    }


def kernel_db_insert(
    project_id: str,
    cell: Dict[str, Any],
    update_best: bool = True,
    default_root: Optional[str] = None,
) -> Dict[str, Any]:
    archive = read_archive(project_id, default_root)
    cid = cell.get("id") or f"cell_{int(time.time() * 1000)}"
    cell["id"] = cid
    if "inserted_at" not in cell:
        cell["inserted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    archive.setdefault("cells", {})[cid] = cell
    if update_best:
        update_global_best(archive)

    db_file = get_db_path(project_id, default_root)
    write_archive_atomic(db_file, archive)
    return {
        "status": "inserted",
        "project_id": project_id,
        "cell_id": cid,
        "total_cells": len(archive["cells"]),
        "global_best_id": archive.get("global_best_id"),
    }


def kernel_db_status(
    project_id: str,
    default_root: Optional[str] = None,
) -> Dict[str, Any]:
    archive = read_archive(project_id, default_root)
    cells = archive.get("cells", {})
    best_id = archive.get("global_best_id")
    best_cell = cells.get(best_id) if best_id else None

    return {
        "project_id": project_id,
        "target_hardware": archive.get("target_hardware"),
        "total_cells": len(cells),
        "global_best_id": best_id,
        "global_best_latency_ms": best_cell.get("fitness_latency_ms")
        if best_cell
        else None,
        "global_best_tflops": best_cell.get("fitness_tflops") if best_cell else None,
        "dimensions": archive.get("dimensions"),
        "bins_per_dimension": archive.get("bins_per_dimension"),
    }


# --- MCP Tool Schemas ---

TOOLS = [
    {
        "name": "kernel_db_init",
        "description": "Initialize a MaxEvolve kernel project directory with MAP-Elites database, artifacts/, candidates/, and source/ layout.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Unique project slug (e.g. vector_add_v1)",
                },
                "target_hardware": {
                    "type": "string",
                    "description": "cpu_interpret | gpu_hopper | tpu_v5e | tpu_v6e",
                },
                "default_root": {
                    "type": "string",
                    "description": "Optional custom project directory root",
                },
            },
            "required": ["project_id", "target_hardware"],
        },
    },
    {
        "name": "kernel_db_sample",
        "description": "Sample candidate elite kernels from the MAP-Elites archive by strategy (fastest, curiosity, random).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project slug"},
                "strategy": {
                    "type": "string",
                    "description": "fastest | curiosity | random",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Number of candidates to sample",
                },
                "mutation_class": {
                    "type": "string",
                    "description": "Optional filter by mutation class",
                },
            },
            "required": ["project_id", "strategy"],
        },
    },
    {
        "name": "kernel_db_insert",
        "description": "Insert or update an evaluated kernel candidate into the MAP-Elites grid.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project slug"},
                "cell": {
                    "type": "object",
                    "description": "Candidate cell object containing id, kernel_path, fitness_latency_ms, etc.",
                },
                "update_best": {
                    "type": "boolean",
                    "description": "Whether to recompute global best",
                },
            },
            "required": ["project_id", "cell"],
        },
    },
    {
        "name": "kernel_db_status",
        "description": "Get current MAP-Elites archive statistics and global best kernel performance for a project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project slug"}
            },
            "required": ["project_id"],
        },
    },
]


def handle_call(name: str, args: Dict[str, Any]) -> str:
    if name == "kernel_db_init":
        res = kernel_db_init(
            args["project_id"],
            args["target_hardware"],
            args.get("default_root"),
        )
    elif name == "kernel_db_sample":
        res = kernel_db_sample(
            args["project_id"],
            args["strategy"],
            args.get("batch_size", 1),
            args.get("mutation_class"),
        )
    elif name == "kernel_db_insert":
        res = kernel_db_insert(
            args["project_id"],
            args["cell"],
            args.get("update_best", True),
        )
    elif name == "kernel_db_status":
        res = kernel_db_status(args["project_id"])
    else:
        raise ValueError(f"Unknown tool: {name}")

    return json.dumps(res, indent=2)


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "kernel-db", "version": "1.0.0"},
                    "capabilities": {"tools": {}},
                },
            }
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                content = handle_call(tool_name, tool_args)
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": content}],
                        "isError": False,
                    },
                }
            except Exception as e:
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                        "isError": True,
                    },
                }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
