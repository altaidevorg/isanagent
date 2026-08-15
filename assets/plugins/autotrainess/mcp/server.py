#!/usr/bin/env python3
"""MCP Stdio Server for AutoTrainess (Experiment Ledger Tools)."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def get_project_root(project_id: str, default_root: Optional[str] = None) -> Path:
    base = default_root or os.environ.get("ISANAGENT_TRAIN_ROOT") or "train/projects"
    return Path(base) / project_id.strip()


def get_db_path(project_id: str, default_root: Optional[str] = None) -> Path:
    return (
        get_project_root(project_id, default_root)
        / "database"
        / "experiment_ledger.json"
    )


def read_ledger(project_id: str, default_root: Optional[str] = None) -> Dict[str, Any]:
    db_file = get_db_path(project_id, default_root)
    if not db_file.exists():
        raise FileNotFoundError(
            f"Experiment ledger not found at {db_file}. Run train_db_init first."
        )
    with open(db_file, "r", encoding="utf-8") as f:
        return json.load(f)


def write_ledger_atomic(path: Path, ledger: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    tmp.replace(path)


# --- Tool Implementations ---


def train_db_init(
    project_id: str,
    model_name: str,
    training_type: str,
    target_hardware: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    default_root: Optional[str] = None,
) -> Dict[str, Any]:
    root = get_project_root(project_id, default_root)
    (root / "database").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "eval").mkdir(parents=True, exist_ok=True)

    db_file = root / "database" / "experiment_ledger.json"
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "model_name": model_name,
        "training_type": training_type,
        "target_hardware": target_hardware,
        "hyperparams": hyperparams or {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "initialized",
        "entries": [],
    }
    write_ledger_atomic(db_file, ledger)
    return {
        "status": "initialized",
        "project_id": project_id,
        "model_name": model_name,
        "training_type": training_type,
        "project_root": str(root),
        "db_path": str(db_file),
    }


def train_db_append(
    project_id: str,
    step: int,
    metrics: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    notes: Optional[str] = None,
    default_root: Optional[str] = None,
) -> Dict[str, Any]:
    ledger = read_ledger(project_id, default_root)
    entry = {
        "step": step,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
        "notes": notes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    ledger.setdefault("entries", []).append(entry)
    ledger["status"] = "in_progress"

    db_file = get_db_path(project_id, default_root)
    write_ledger_atomic(db_file, ledger)
    return {
        "status": "recorded",
        "project_id": project_id,
        "step": step,
        "total_entries": len(ledger["entries"]),
    }


def train_db_list(default_root: Optional[str] = None) -> Dict[str, Any]:
    base = Path(
        default_root or os.environ.get("ISANAGENT_TRAIN_ROOT") or "train/projects"
    )
    if not base.exists():
        return {"projects": []}

    projects = []
    for entry in base.iterdir():
        if entry.is_dir():
            db = entry / "database" / "experiment_ledger.json"
            if db.exists():
                try:
                    with open(db, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    projects.append(
                        {
                            "project_id": data.get("project_id", entry.name),
                            "model_name": data.get("model_name"),
                            "training_type": data.get("training_type"),
                            "status": data.get("status"),
                            "total_steps": len(data.get("entries", [])),
                        }
                    )
                except Exception:
                    projects.append(
                        {"project_id": entry.name, "status": "unreadable"}
                    )
    return {"projects": projects}


def train_db_status(
    project_id: str, default_root: Optional[str] = None
) -> Dict[str, Any]:
    ledger = read_ledger(project_id, default_root)
    entries = ledger.get("entries", [])
    latest_entry = entries[-1] if entries else None

    return {
        "project_id": project_id,
        "model_name": ledger.get("model_name"),
        "training_type": ledger.get("training_type"),
        "target_hardware": ledger.get("target_hardware"),
        "status": ledger.get("status"),
        "total_steps_logged": len(entries),
        "latest_entry": latest_entry,
        "created_at": ledger.get("created_at"),
    }


def train_db_get(
    project_id: str,
    step: Optional[int] = None,
    default_root: Optional[str] = None,
) -> Dict[str, Any]:
    ledger = read_ledger(project_id, default_root)
    if step is not None:
        matched = [e for e in ledger.get("entries", []) if e.get("step") == step]
        if not matched:
            raise KeyError(f"Step {step} not found for project {project_id}")
        return {"project_id": project_id, "step_entry": matched[0]}

    return ledger


# --- MCP Tool Schemas ---

TOOLS = [
    {
        "name": "train_db_init",
        "description": "Initialize a new AutoTrainess training project and experiment ledger.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Unique training project slug",
                },
                "model_name": {
                    "type": "string",
                    "description": "Base model identifier (e.g. Qwen/Qwen2.5-Coder-7B)",
                },
                "training_type": {
                    "type": "string",
                    "description": "sft | dpo | ppo | pretrain | lora",
                },
                "target_hardware": {
                    "type": "string",
                    "description": "Target GPU/cluster environment (e.g. 8xH100, RTX4090)",
                },
                "hyperparams": {
                    "type": "object",
                    "description": "Optional initial hyperparameters dictionary",
                },
                "default_root": {
                    "type": "string",
                    "description": "Optional project root directory",
                },
            },
            "required": [
                "project_id",
                "model_name",
                "training_type",
                "target_hardware",
            ],
        },
    },
    {
        "name": "train_db_append",
        "description": "Log an evaluation or training step outcome to the experiment ledger.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project slug"},
                "step": {
                    "type": "integer",
                    "description": "Training or iteration step index",
                },
                "metrics": {
                    "type": "object",
                    "description": "Metrics map (e.g. train_loss, eval_loss, accuracy, mfu)",
                },
                "checkpoint_path": {
                    "type": "string",
                    "description": "Saved checkpoint path if any",
                },
                "notes": {
                    "type": "string",
                    "description": "Operator or subagent qualitative notes",
                },
            },
            "required": ["project_id", "step", "metrics"],
        },
    },
    {
        "name": "train_db_list",
        "description": "List all registered AutoTrainess projects and their high-level statuses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "default_root": {
                    "type": "string",
                    "description": "Optional root directory",
                }
            },
        },
    },
    {
        "name": "train_db_status",
        "description": "Get status, latest step, and metrics summary for an AutoTrainess project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project slug"}
            },
            "required": ["project_id"],
        },
    },
    {
        "name": "train_db_get",
        "description": "Fetch full experiment ledger history or a specific step entry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project slug"},
                "step": {
                    "type": "integer",
                    "description": "Optional specific step index to retrieve",
                },
            },
            "required": ["project_id"],
        },
    },
]


def handle_call(name: str, args: Dict[str, Any]) -> str:
    if name == "train_db_init":
        res = train_db_init(
            args["project_id"],
            args["model_name"],
            args["training_type"],
            args["target_hardware"],
            args.get("hyperparams"),
            args.get("default_root"),
        )
    elif name == "train_db_append":
        res = train_db_append(
            args["project_id"],
            args["step"],
            args["metrics"],
            args.get("checkpoint_path"),
            args.get("notes"),
            args.get("default_root"),
        )
    elif name == "train_db_list":
        res = train_db_list(args.get("default_root"))
    elif name == "train_db_status":
        res = train_db_status(
            args["project_id"],
            args.get("default_root"),
        )
    elif name == "train_db_get":
        res = train_db_get(
            args["project_id"],
            args.get("step"),
            args.get("default_root"),
        )
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
                    "serverInfo": {"name": "train-db", "version": "1.0.0"},
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
