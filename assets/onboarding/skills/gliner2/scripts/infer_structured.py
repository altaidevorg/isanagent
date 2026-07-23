# /// script
# requires-python = ">=3.10"
# dependencies = ["gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"]
# ///
"""Run local GLiNER2 structured extraction with public schema preflight.

Examples:
    uv run infer_structured.py --preflight
    uv run infer_structured.py --smoke-test
    uv run infer_structured.py --text-file invoice.txt --schema-file schema.json
"""

from __future__ import annotations

import argparse
import contextlib
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import sys
from typing import Any


SMOKE_TEXT = (
    "Invoice INV-2026-041 from Northwind Supplies to Contoso Retail. "
    "Issued on 2026-07-10 and due on 2026-08-10. Currency: USD. "
    "Payment status: unpaid. Line items: 2 keyboards at USD 75 each, total USD 150; "
    "3 mice at USD 20 each, total USD 60. Invoice total: USD 210."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local GLiNER2 structured extraction and emit validated JSON."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to process with the selected schema.")
    source.add_argument("--text-file", type=Path, help="UTF-8 text file to process.")
    parser.add_argument(
        "--schema-file",
        type=Path,
        help="Optional user-friendly Schema.from_dict JSON file; defaults to invoice + line_item.",
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
        "--preflight",
        action="store_true",
        help="Validate and print the public schema without loading model weights.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a bounded synthetic invoice and deterministic arithmetic checks.",
    )
    return parser


def default_schema(Schema: Any) -> Any:
    # Keep repeated records as flat siblings. A targeted threshold avoids lowering
    # every field merely to recover weak line-item totals.
    return (
        Schema()
        .structure("invoice")
            .field("invoice_number", dtype="str", description="Invoice identifier")
            .field("vendor_name", dtype="str", description="Seller or issuing company")
            .field("customer_name", dtype="str", description="Buyer or billed company")
            .field("issue_date", dtype="str")
            .field("due_date", dtype="str")
            .field("currency", dtype="str")
            .field(
                "payment_status",
                dtype="str",
                choices=["paid", "unpaid", "partial", "overdue"],
            )
            .field("invoice_total", dtype="str")
        .structure("line_item")
            .field("description", dtype="str")
            .field("quantity", dtype="str")
            .field("unit_price", dtype="str")
            .field("total", dtype="str", threshold=0.15)
    )


def load_schema(schema_path: Path | None, Schema: Any) -> Any:
    if schema_path is None:
        return default_schema(Schema)
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    return Schema.from_dict(data)


def preflight_schema(schema: Any, Schema: Any) -> dict[str, Any]:
    friendly = schema.to_dict()
    # Public validation round trip. Keep using the original builder afterwards so
    # advanced thresholds/validators not represented by to_dict remain active.
    Schema.from_dict(friendly)
    built = schema.build()
    structures = friendly.get("structures", {})
    issues: list[str] = []
    for structure_name, structure in structures.items():
        fields = structure.get("fields", [])
        if not fields:
            issues.append(f"{structure_name}: no fields")
        names = [field.get("name") for field in fields]
        if len(names) != len(set(names)):
            issues.append(f"{structure_name}: duplicate field names")
    if not built:
        issues.append("schema.build() returned an empty mapping")
    return {
        "ok": not issues,
        "issues": issues,
        "schema": friendly,
        "inference_schema_sections": sorted(built),
    }


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
    """Recursively validate span dictionaries; choice values may omit offsets."""
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


def unbox(value: Any) -> Any:
    if isinstance(value, dict) and "text" in value:
        return value["text"]
    return value


def as_text(value: Any) -> str | None:
    value = unbox(value)
    return value if isinstance(value, str) and value.strip() else None


def decimal_value(value: Any) -> Decimal:
    text = as_text(value)
    if text is None:
        raise ValueError("missing numeric value")
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"no numeric value in {text!r}")
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError(f"invalid numeric value in {text!r}") from exc


def smoke_business_checks(result: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    invoices = result.get("invoice", [])
    line_items = result.get("line_item", [])
    if len(invoices) != 1:
        issues.append(f"expected 1 invoice record, received {len(invoices)}")
    if len(line_items) != 2:
        issues.append(f"expected 2 line-item records, received {len(line_items)}")

    required_invoice = (
        "invoice_number", "vendor_name", "customer_name", "issue_date", "due_date",
        "currency", "payment_status", "invoice_total",
    )
    if invoices:
        invoice = invoices[0]
        for field in required_invoice:
            if as_text(invoice.get(field)) is None:
                issues.append(f"invoice.{field} is missing")
        expected = {
            "invoice_number": "INV-2026-041",
            "vendor_name": "Northwind Supplies",
            "customer_name": "Contoso Retail",
            "issue_date": "2026-07-10",
            "due_date": "2026-08-10",
            "currency": "USD",
            "payment_status": "unpaid",
        }
        for field, expected_value in expected.items():
            actual = as_text(invoice.get(field))
            if actual != expected_value:
                issues.append(f"invoice.{field}: expected {expected_value!r}, received {actual!r}")

    computed_totals: list[Decimal] = []
    for index, item in enumerate(line_items):
        for field in ("description", "quantity", "unit_price", "total"):
            if as_text(item.get(field)) is None:
                issues.append(f"line_item[{index}].{field} is missing")
        try:
            quantity = decimal_value(item.get("quantity"))
            unit_price = decimal_value(item.get("unit_price"))
            total = decimal_value(item.get("total"))
            computed_totals.append(total)
            if quantity * unit_price != total:
                issues.append(
                    f"line_item[{index}] arithmetic failed: {quantity} * {unit_price} != {total}"
                )
        except ValueError as exc:
            issues.append(f"line_item[{index}] arithmetic unavailable: {exc}")

    if invoices and len(computed_totals) == len(line_items) == 2:
        try:
            invoice_total = decimal_value(invoices[0].get("invoice_total"))
            if sum(computed_totals, Decimal("0")) != invoice_total:
                issues.append(
                    f"invoice total failed: line items sum to {sum(computed_totals, Decimal('0'))}, "
                    f"invoice says {invoice_total}"
                )
        except ValueError as exc:
            issues.append(f"invoice total unavailable: {exc}")
    return {"ok": not issues, "issues": issues}


def package_report(gliner2_module: Any, extractor: Any, requested_device: str) -> dict[str, Any]:
    return {
        "version": getattr(gliner2_module, "__version__", None),
        "module_path": str(Path(gliner2_module.__file__).resolve()),
        "python": sys.executable,
        "requested_device": requested_device,
        "parameter_devices": sorted({str(p.device) for p in extractor.parameters()}),
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.smoke_test and (args.text is not None or args.text_file is not None or args.schema_file):
        raise ValueError("--smoke-test cannot be combined with custom text or schema")

    import gliner2
    from gliner2 import GLiNER2, Schema

    schema = load_schema(args.schema_file, Schema)
    preflight = preflight_schema(schema, Schema)
    if args.preflight:
        report = {
            "status": "PASS" if preflight["ok"] else "FAIL",
            "mode": "preflight",
            "environment": {
                "version": getattr(gliner2, "__version__", None),
                "module_path": str(Path(gliner2.__file__).resolve()),
                "python": sys.executable,
            },
            "preflight": preflight,
        }
        return report, 0 if preflight["ok"] else 1

    if args.smoke_test:
        source = SMOKE_TEXT
    elif args.text_file is not None:
        source = args.text_file.read_text(encoding="utf-8")
    elif args.text is not None:
        source = args.text
    else:
        raise ValueError("Provide --text, --text-file, --smoke-test, or --preflight")

    device = resolve_device(args.device)
    with contextlib.redirect_stdout(sys.stderr):
        extractor = GLiNER2.from_pretrained(args.model, map_location=device)
        extractor.eval()
        result = extractor.extract(
            source,
            schema,
            threshold=args.threshold,
            include_confidence=True,
            include_spans=True,
        )

    span_count, span_errors = validate_spans(result, source)
    expected_structures = set(preflight["schema"].get("structures", {}))
    shape_issues = [
        f"{name}: missing list result"
        for name in sorted(expected_structures)
        if not isinstance(result.get(name), list)
    ]
    business = smoke_business_checks(result) if args.smoke_test else None

    status = "PASS"
    if not preflight["ok"] or span_errors or shape_issues:
        status = "FAIL"
    elif business is not None and not business["ok"]:
        status = "PARTIAL"

    report = {
        "status": status,
        "mode": "smoke_test" if args.smoke_test else "inference",
        "model": args.model,
        "environment": package_report(gliner2, extractor, device),
        "request": {
            "text": source,
            "threshold": args.threshold,
            "include_confidence": True,
            "include_spans": True,
        },
        "preflight": preflight,
        "result": result,
        "validation": {
            "span_objects_checked": span_count,
            "span_errors": span_errors,
            "shape_issues": shape_issues,
            "smoke_business_checks": business,
        },
    }
    return report, 0 if status == "PASS" else (2 if status == "PARTIAL" else 1)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
