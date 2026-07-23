#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"]
# ///
"""Validate GLiNER2 JSONL before model loading or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate GLiNER2 JSONL, relation/classification schema "
            "consistency, and cross-file text leakage."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL file(s) to validate")
    parser.add_argument("--report", type=Path, help="Also write the JSON report to this path")
    parser.add_argument(
        "--allow-cross-file-overlap",
        action="store_true",
        help="Allow identical normalized text in different input files",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Treat duplicate records within one file as errors instead of warnings",
    )
    return parser.parse_args()


def canonical_record(raw: dict[str, Any], source: str, line: int) -> dict[str, Any]:
    if "input" in raw and "output" in raw:
        return {"input": raw["input"], "output": raw["output"]}
    if "text" in raw and "schema" in raw:
        return {"input": raw["text"], "output": raw["schema"]}
    raise ValueError(f"{source}:{line}: expected input/output or text/schema")


def read_jsonl(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            errors.append(f"{path}: file not found")
            continue
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("top-level JSON value must be an object")
                    record = canonical_record(raw, str(path), line_no)
                    records.append(record)
                    provenance.append({"path": str(path.resolve()), "line": line_no})
                except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
                    errors.append(f"{path}:{line_no}: {exc}")
    return records, provenance, errors


def normalized_text(text: Any) -> str:
    return " ".join(str(text).split()).casefold()


def record_digest(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def duplicate_checks(
    records: list[dict[str, Any]], provenance: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record_locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    text_locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, origin in zip(records, provenance):
        record_locations[record_digest(record)].append(origin)
        text_locations[normalized_text(record.get("input", ""))].append(origin)

    duplicate_records = [
        {"sha256": digest, "locations": locations}
        for digest, locations in record_locations.items()
        if len(locations) > 1
    ]
    cross_file_overlap = []
    for text_key, locations in text_locations.items():
        if text_key and len({item["path"] for item in locations}) > 1:
            cross_file_overlap.append(
                {
                    "text_sha256": hashlib.sha256(text_key.encode("utf-8")).hexdigest(),
                    "locations": locations,
                }
            )
    return duplicate_records, cross_file_overlap


def classification_schema_errors(dataset: Any) -> list[str]:
    schemas: dict[str, tuple[tuple[str, ...], bool, int]] = {}
    errors: list[str] = []
    for index, example in enumerate(dataset):
        for task in example.classifications:
            signature = (tuple(task.labels), bool(task.multi_label))
            if task.task in schemas:
                labels, multi_label, first_index = schemas[task.task]
                if signature != (labels, multi_label):
                    errors.append(
                        f"Classification task '{task.task}' differs between examples "
                        f"{first_index} and {index}: labels/order or multi_label changed"
                    )
            else:
                schemas[task.task] = (*signature, index)
    return errors


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    records, provenance, parse_errors = read_jsonl(args.paths)
    duplicate_records, cross_file_overlap = duplicate_checks(records, provenance)

    validation_report: dict[str, Any] = {
        "valid": 0,
        "invalid": len(parse_errors),
        "total": len(records) + len(parse_errors),
        "invalid_indices": [],
        "errors": list(parse_errors),
    }
    relation_errors: list[str] = []
    classification_errors: list[str] = []
    stats: dict[str, Any] = {}

    if records:
        from gliner2.training.data import TrainingDataset

        try:
            dataset = TrainingDataset.from_records(records)
            # Current GLiNER2 validation is always strict. It has no strict= keyword.
            validation_report = dataset.validate(raise_on_error=False)
            validation_report["errors"] = parse_errors + validation_report["errors"]
            validation_report["invalid"] += len(parse_errors)
            validation_report["total"] += len(parse_errors)
            relation_errors = dataset.validate_relation_consistency()
            classification_errors = classification_schema_errors(dataset)
            stats = dataset.stats()
        except Exception as exc:
            validation_report["errors"].append(f"Dataset parsing failed: {exc}")
            validation_report["invalid"] = max(1, validation_report["invalid"])
    else:
        validation_report["errors"].append("Dataset is empty: no valid nonblank records were found")
        validation_report["invalid"] = max(1, validation_report["invalid"])
        validation_report["total"] = max(1, validation_report["total"])

    hard_errors = bool(validation_report["errors"] or relation_errors or classification_errors)
    if cross_file_overlap and not args.allow_cross_file_overlap:
        hard_errors = True
    if duplicate_records and args.fail_on_duplicates:
        hard_errors = True

    report = {
        "status": "FAIL" if hard_errors else "PASS",
        "files": [str(path.resolve()) for path in args.paths],
        "validation": validation_report,
        "relation_consistency_errors": relation_errors,
        "classification_schema_errors": classification_errors,
        "duplicate_records": duplicate_records,
        "cross_file_text_overlap": cross_file_overlap,
        "duplicates_are_errors": bool(args.fail_on_duplicates),
        "cross_file_overlap_allowed": bool(args.allow_cross_file_overlap),
        "statistics": stats,
    }
    rendered = json.dumps(json_safe(report), ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 1 if hard_errors else 0


if __name__ == "__main__":
    sys.exit(main())
