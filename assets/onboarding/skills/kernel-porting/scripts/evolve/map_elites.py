# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""MAP-Elites archive helpers (used by evolve_runner and agents)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EliteCell:
    id: str
    kernel_path: str
    fitness_latency_ms: float | None = None
    fitness_mfu: float | None = None
    fitness_tflops: float | None = None
    complexity_loc: int | None = None
    complexity_ast_depth: int | None = None
    tile_volume: int | None = None
    mutation_class: str | None = None
    parent_id: str | None = None
    generation: int | None = None
    notes: str | None = None
    inserted_at: str | None = None

    def map_key(self) -> str:
        return f"{self.complexity_loc or 0}:{self.tile_volume or 0}"


def load_archive(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_archive(path: Path, archive: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(archive, indent=2), encoding="utf-8")
    tmp.replace(path)


def insert_cell(archive: dict[str, Any], cell: EliteCell) -> bool:
    if cell.inserted_at is None:
        cell.inserted_at = datetime.now(timezone.utc).isoformat()
    key = cell.map_key()
    cells = archive.setdefault("cells", {})
    existing = cells.get(key)
    cell_lat = cell.fitness_latency_ms if cell.fitness_latency_ms is not None else 1e9
    exist_lat = existing.get("fitness_latency_ms") if existing and existing.get("fitness_latency_ms") is not None else 1e9
    if existing is None or cell_lat < exist_lat:
        cells[key] = asdict(cell)
        replaced = True
    else:
        replaced = False
    best = None
    best_lat = float("inf")
    for c in cells.values():
        lat = c.get("fitness_latency_ms")
        if lat is not None and lat < best_lat:
            best_lat = lat
            best = c.get("id")
    archive["global_best_id"] = best
    return replaced


def sample_top(archive: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
    cells = list(archive.get("cells", {}).values())
    cells.sort(key=lambda c: c.get("fitness_latency_ms") if c.get("fitness_latency_ms") is not None else 1e18)
    return cells[:k]
