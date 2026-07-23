# /// script
# requires-python = ">=3.10"
# dependencies = ["gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"]
# ///
"""Run bounded local GLiNER2 entity extraction and verify source spans.

Examples:
    uv run infer_entities.py --smoke-test
    uv run infer_entities.py --text "Ada joined Acme." \
        --label person="Names of people" --label company="Organizations"
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
from typing import Any


SMOKE_TEXT = "Apple CEO Tim Cook introduced Vision Pro in Cupertino."
SMOKE_LABELS = {
    "company": "Companies and organizations",
    "person": "Names of people",
    "product": "Commercial products",
    "location": "Cities and physical locations",
}
SMOKE_EXPECTED = {
    "company": "Apple",
    "person": "Tim Cook",
    "product": "Vision Pro",
    "location": "Cupertino",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local GLiNER2 NER, emit JSON, and validate every returned span."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to process.")
    source.add_argument("--text-file", type=Path, help="UTF-8 text file to process.")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="NAME[=DESCRIPTION]",
        help="Entity label; repeat for multiple labels.",
    )
    parser.add_argument(
        "--model", default="fastino/gliner2-base-v1", help="Model ID or local model directory."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
        help="Execution device. The conservative default is cpu.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one bounded synthetic example and verify expected mentions.",
    )
    return parser


def parse_labels(values: list[str]) -> list[str] | dict[str, str]:
    parsed: list[tuple[str, str | None]] = []
    for value in values:
        name, separator, description = value.partition("=")
        name = name.strip()
        if not name:
            raise ValueError("Entity label names cannot be empty")
        parsed.append((name, description.strip() if separator else None))
    if len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("Entity label names must be unique")
    if any(description is not None for _, description in parsed):
        return {
            name: description or name.replace("_", " ")
            for name, description in parsed
        }
    return [name for name, _ in parsed]


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def validate_spans(value: Any, source: str, path: str = "$", errors: list[str] | None = None) -> tuple[int, list[str]]:
    """Recursively validate every dict that carries start/end offsets."""
    if errors is None:
        errors = []
    count = 0
    if isinstance(value, dict):
        has_start = "start" in value
        has_end = "end" in value
        if has_start or has_end:
            count += 1
            if not (has_start and has_end and "text" in value):
                errors.append(f"{path}: span object must contain text, start, and end")
            else:
                start, end, text = value["start"], value["end"], value["text"]
                if not isinstance(start, int) or not isinstance(end, int):
                    errors.append(f"{path}: start/end must be integers")
                elif not (0 <= start <= end <= len(source)):
                    errors.append(f"{path}: offsets [{start}:{end}] are outside the source")
                elif source[start:end] != text:
                    errors.append(
                        f"{path}: source[{start}:{end}]={source[start:end]!r}, output text={text!r}"
                    )
        for key, item in value.items():
            child_count, _ = validate_spans(item, source, f"{path}.{key}", errors)
            count += child_count
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_count, _ = validate_spans(item, source, f"{path}[{index}]", errors)
            count += child_count
    return count, errors


def entity_texts(result: dict[str, Any], label: str) -> list[str]:
    values = result.get("entities", {}).get(label, [])
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    texts = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("text")
        if isinstance(value, str):
            texts.append(value)
    return texts


def package_report(gliner2_module: Any, extractor: Any, requested_device: str) -> dict[str, Any]:
    devices = sorted({str(parameter.device) for parameter in extractor.parameters()})
    return {
        "version": getattr(gliner2_module, "__version__", None),
        "module_path": str(Path(gliner2_module.__file__).resolve()),
        "python": sys.executable,
        "requested_device": requested_device,
        "parameter_devices": devices,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")

    if args.smoke_test:
        if args.text is not None or args.text_file is not None or args.label:
            raise ValueError("--smoke-test cannot be combined with custom text or labels")
        source = SMOKE_TEXT
        labels: list[str] | dict[str, str] = SMOKE_LABELS
    else:
        if args.text_file is not None:
            source = args.text_file.read_text(encoding="utf-8")
        elif args.text is not None:
            source = args.text
        else:
            raise ValueError("Provide --text, --text-file, or --smoke-test")
        if not args.label:
            raise ValueError("Provide at least one --label for custom inference")
        labels = parse_labels(args.label)

    device = resolve_device(args.device)
    import gliner2
    from gliner2 import GLiNER2

    # Keep stdout as one machine-readable JSON document; library progress goes to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        extractor = GLiNER2.from_pretrained(args.model, map_location=device)
        extractor.eval()
        result = extractor.extract_entities(
            source,
            labels,
            threshold=args.threshold,
            include_confidence=True,
            include_spans=True,
        )

    span_count, span_errors = validate_spans(result, source)
    semantic_checks: dict[str, bool] = {}
    if args.smoke_test:
        semantic_checks = {
            label: expected in entity_texts(result, label)
            for label, expected in SMOKE_EXPECTED.items()
        }

    structural_ok = isinstance(result, dict) and isinstance(result.get("entities"), dict)
    status = "PASS"
    if span_errors or not structural_ok:
        status = "FAIL"
    elif semantic_checks and not all(semantic_checks.values()):
        status = "PARTIAL"

    report = {
        "status": status,
        "mode": "smoke_test" if args.smoke_test else "inference",
        "model": args.model,
        "environment": package_report(gliner2, extractor, device),
        "request": {
            "text": source,
            "entity_types": labels,
            "threshold": args.threshold,
            "include_confidence": True,
            "include_spans": True,
        },
        "result": result,
        "validation": {
            "structural_ok": structural_ok,
            "span_objects_checked": span_count,
            "span_errors": span_errors,
            "smoke_expected_mentions": semantic_checks,
        },
    }
    return report, 0 if status == "PASS" else (2 if status == "PARTIAL" else 1)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report, code = run(args)
    except Exception as exc:  # Emit a stable error envelope for CLI automation.
        report = {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        code = 1
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
