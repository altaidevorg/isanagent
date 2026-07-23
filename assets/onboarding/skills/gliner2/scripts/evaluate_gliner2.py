#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"]
# ///
"""Task-aware GLiNER2 evaluation with machine-readable output."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more GLiNER2 models on gold JSONL, or score an aligned "
            "prediction JSONL without loading a model."
        )
    )
    parser.add_argument("--data", required=True, type=Path, help="Gold GLiNER2 JSONL")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        action="append",
        help="Model ID/path; repeat to compare baseline and fine-tuned models. Use NAME=PATH to label.",
    )
    group.add_argument(
        "--predictions",
        type=Path,
        help="Aligned JSONL containing a top-level prediction object per gold row",
    )
    parser.add_argument("--output", type=Path, help="Write the complete JSON report here")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-examples", type=int, default=-1)
    return parser.parse_args()


def canonical_record(raw: dict[str, Any], source: Path, line_no: int) -> dict[str, Any]:
    if "input" in raw and "output" in raw:
        return {"input": raw["input"], "output": raw["output"]}
    if "text" in raw and "schema" in raw:
        return {"input": raw["text"], "output": raw["schema"]}
    raise ValueError(f"{source}:{line_no}: expected input/output or text/schema")


def load_gold(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            records.append(canonical_record(raw, path, line_no))
            if limit > 0 and len(records) >= limit:
                break

    if not records:
        raise ValueError(f"Gold dataset is empty: {path}")

    from gliner2.training.data import TrainingDataset

    dataset = TrainingDataset.from_records(records)
    report = dataset.validate(raise_on_error=False)
    relation_errors = dataset.validate_relation_consistency()
    if report["invalid"] or relation_errors:
        raise ValueError("Gold data failed validation: " + "; ".join(report["errors"] + relation_errors))
    return records


def load_predictions(path: Path, expected: int) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or not isinstance(raw.get("prediction"), dict):
                raise ValueError(f"{path}:{line_no}: expected a top-level prediction object")
            predictions.append(raw["prediction"])
    if len(predictions) != expected:
        raise ValueError(f"Prediction count {len(predictions)} does not match gold count {expected}")
    return predictions


def norm_text(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        if "value" in value and "choices" in value:
            return unwrap(value["value"])
        if "text" in value:
            return unwrap(value["text"])
        return {key: unwrap(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unwrap(item) for item in value]
    if isinstance(value, tuple):
        return [unwrap(item) for item in value]
    return value


def norm_value(value: Any) -> Any:
    value = unwrap(value)
    if value is None:
        return None
    if isinstance(value, str):
        return norm_text(value) if value.strip() else None
    if isinstance(value, list):
        items = [norm_value(item) for item in value]
        return sorted((item for item in items if item is not None), key=lambda item: repr(item))
    if isinstance(value, dict):
        return {
            key: normalized
            for key, item in value.items()
            if (normalized := norm_value(item)) is not None
        }
    return value


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def entity_tuples(mapping: Any) -> set[tuple[str, str]]:
    output: set[tuple[str, str]] = set()
    if not isinstance(mapping, dict):
        return output
    for label, values in mapping.items():
        values = values if isinstance(values, list) else [values]
        for value in values:
            text = norm_value(value)
            if isinstance(text, str):
                output.add((label, text))
    return output


def relation_tuples(mapping: Any) -> set[tuple[str, str, str]]:
    output: set[tuple[str, str, str]] = set()
    if not isinstance(mapping, dict):
        return output
    for relation, values in mapping.items():
        values = values if isinstance(values, list) else [values]
        for value in values:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                head, tail = value
            elif isinstance(value, dict) and "head" in value and "tail" in value:
                head, tail = value["head"], value["tail"]
            else:
                continue
            output.add((relation, norm_text(unwrap(head)), norm_text(unwrap(tail))))
    return output


def gold_relations(output: dict[str, Any]) -> tuple[set[tuple[str, str, str]], list[dict[str, Any]]]:
    supported: set[tuple[str, str, str]] = set()
    unsupported: list[dict[str, Any]] = []
    for relation_record in output.get("relations", []):
        for relation, fields in relation_record.items():
            if set(fields) == {"head", "tail"}:
                supported.add((relation, norm_text(fields["head"]), norm_text(fields["tail"])))
            else:
                unsupported.append({"relation": relation, "fields": sorted(fields)})
    return supported, unsupported


def classification_predictions(prediction: dict[str, Any]) -> dict[str, Any]:
    value = prediction.get("classifications", {})
    return value if isinstance(value, dict) else {}


def structure_schema(output: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    fields: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in output.get("json_structures", []):
        for parent, values in item.items():
            for name, value in values.items():
                if name in fields[parent]:
                    continue
                if isinstance(value, dict) and "choices" in value:
                    fields[parent][name] = {
                        "name": name,
                        "dtype": "str",
                        "choices": value["choices"],
                    }
                else:
                    fields[parent][name] = {
                        "name": name,
                        "dtype": "list" if isinstance(value, list) else "str",
                    }
    return {parent: list(specs.values()) for parent, specs in fields.items()}


def predict_with_model(model: Any, record: dict[str, Any], threshold: float) -> tuple[dict[str, Any], list[str]]:
    text = record["input"]
    output = record["output"]
    prediction: dict[str, Any] = {
        "entities": {}, "classifications": {}, "relations": {}, "structures": {}
    }
    errors: list[str] = []

    if "entities" in output and output["entities"]:
        descriptions = output.get("entity_descriptions", {})
        entity_types = {name: descriptions.get(name, "") for name in output["entities"]}
        try:
            result = model.extract_entities(text, entity_types, threshold=threshold)
            prediction["entities"] = result.get("entities", {})
        except Exception as exc:
            errors.append(f"entities: {type(exc).__name__}: {exc}")

    if output.get("classifications"):
        tasks = {}
        for task in output["classifications"]:
            config = {
                "labels": task["labels"],
                "multi_label": task.get("multi_label", len(task.get("true_label", [])) > 1),
            }
            for key in ("prompt", "examples", "label_descriptions"):
                if key in task:
                    config[key] = task[key]
            tasks[task["task"]] = config
        try:
            result = model.classify_text(text, tasks, threshold=threshold)
            prediction["classifications"] = {name: result.get(name) for name in tasks}
        except Exception as exc:
            errors.append(f"classifications: {type(exc).__name__}: {exc}")

    structs = structure_schema(output)
    if structs:
        try:
            prediction["structures"] = model.extract_json(text, structs, threshold=threshold)
        except Exception as exc:
            errors.append(f"structures: {type(exc).__name__}: {exc}")

    supported_relations, _ = gold_relations(output)
    relation_names = sorted({item[0] for item in supported_relations})
    if relation_names:
        try:
            result = model.extract_relations(text, relation_names, threshold=threshold)
            prediction["relations"] = result.get("relation_extraction", {})
        except Exception as exc:
            errors.append(f"relations: {type(exc).__name__}: {exc}")
    return prediction, errors


def required_structure_instances(output: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in output.get("json_structures", []):
        for parent, fields in item.items():
            normalized = {
                name: norm_value(value)
                for name, value in fields.items()
                if norm_value(value) is not None
            }
            grouped[parent].append(normalized)
    return grouped


def predicted_structure_instances(prediction: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw = prediction.get("structures", {})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(raw, dict):
        return grouped
    for parent, instances in raw.items():
        if parent in {"entities", "relation_extraction"}:
            continue
        instances = instances if isinstance(instances, list) else [instances]
        for fields in instances:
            if isinstance(fields, dict):
                grouped[parent].append(
                    {
                        name: norm_value(value)
                        for name, value in fields.items()
                        if norm_value(value) is not None
                    }
                )
    return grouped


def canonical_counter(grouped: dict[str, list[dict[str, Any]]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for parent, instances in grouped.items():
        for fields in instances:
            counter[json.dumps([parent, fields], sort_keys=True, ensure_ascii=False)] += 1
    return counter


def evaluate(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    api_errors: list[Any],
    *,
    api_executed: bool,
) -> dict[str, Any]:
    entity_gold: set[tuple[int, str, str]] = set()
    entity_pred: set[tuple[int, str, str]] = set()
    relation_gold: set[tuple[int, str, str, str]] = set()
    relation_pred: set[tuple[int, str, str, str]] = set()
    unsupported_relations: list[dict[str, Any]] = []
    single_correct = 0
    single_total = 0
    multi_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    structure_required = 0
    structure_matched = 0
    missing_required: list[dict[str, Any]] = []
    gold_struct_counter: Counter[str] = Counter()
    pred_struct_counter: Counter[str] = Counter()

    for index, (record, prediction) in enumerate(zip(records, predictions)):
        output = record["output"]
        for label, text in entity_tuples(output.get("entities", {})):
            entity_gold.add((index, label, text))
        for label, text in entity_tuples(prediction.get("entities", {})):
            entity_pred.add((index, label, text))

        gold_rel, unsupported = gold_relations(output)
        unsupported_relations.extend({"example": index, **item} for item in unsupported)
        for relation, head, tail in gold_rel:
            relation_gold.add((index, relation, head, tail))
        for relation, head, tail in relation_tuples(prediction.get("relations", {})):
            relation_pred.add((index, relation, head, tail))

        cls_pred = classification_predictions(prediction)
        for task in output.get("classifications", []):
            gold_labels = task["true_label"]
            gold_labels = [gold_labels] if isinstance(gold_labels, str) else list(gold_labels)
            predicted = unwrap(cls_pred.get(task["task"]))
            if isinstance(predicted, dict) and "label" in predicted:
                predicted = predicted["label"]
            predicted_labels = predicted if isinstance(predicted, list) else ([] if predicted is None else [predicted])
            gold_set = {norm_text(label) for label in gold_labels}
            pred_set = {norm_text(label) for label in predicted_labels}
            is_multi = bool(task.get("multi_label", len(gold_labels) > 1))
            if is_multi:
                for label in task["labels"]:
                    key = (task["task"], norm_text(label))
                    in_gold, in_pred = key[1] in gold_set, key[1] in pred_set
                    if in_gold and in_pred:
                        multi_counts[key][0] += 1
                    elif in_pred:
                        multi_counts[key][1] += 1
                    elif in_gold:
                        multi_counts[key][2] += 1
            else:
                single_total += 1
                single_correct += int(len(gold_set) == 1 and pred_set == gold_set)

        gold_grouped = required_structure_instances(output)
        pred_grouped = predicted_structure_instances(prediction)
        gold_struct_counter.update(canonical_counter(gold_grouped))
        pred_struct_counter.update(canonical_counter(pred_grouped))
        for parent, expected_instances in gold_grouped.items():
            available = list(pred_grouped.get(parent, []))
            used: set[int] = set()
            for instance_no, expected in enumerate(expected_instances):
                best_index = None
                best_score = -1
                for candidate_index, candidate in enumerate(available):
                    if candidate_index in used:
                        continue
                    score = sum(candidate.get(field) == value for field, value in expected.items())
                    if score > best_score:
                        best_score, best_index = score, candidate_index
                candidate = available[best_index] if best_index is not None else {}
                if best_index is not None:
                    used.add(best_index)
                for field, expected_value in expected.items():
                    structure_required += 1
                    if candidate.get(field) == expected_value:
                        structure_matched += 1
                    else:
                        missing_required.append(
                            {
                                "example": index,
                                "structure": parent,
                                "instance": instance_no,
                                "field": field,
                                "expected": expected_value,
                                "predicted": candidate.get(field),
                            }
                        )

    entity_tp = len(entity_gold & entity_pred)
    relation_tp = len(relation_gold & relation_pred)
    entity_metrics = prf(entity_tp, len(entity_pred - entity_gold), len(entity_gold - entity_pred))
    relation_metrics = prf(relation_tp, len(relation_pred - relation_gold), len(relation_gold - relation_pred))

    multi_tp = sum(values[0] for values in multi_counts.values())
    multi_fp = sum(values[1] for values in multi_counts.values())
    multi_fn = sum(values[2] for values in multi_counts.values())
    per_label_f1 = [prf(*values)["f1"] for values in multi_counts.values()]
    macro_values = [value for value in per_label_f1 if value is not None]

    struct_intersection = gold_struct_counter & pred_struct_counter
    struct_tp = sum(struct_intersection.values())
    struct_metrics = prf(
        struct_tp,
        sum((pred_struct_counter - gold_struct_counter).values()),
        sum((gold_struct_counter - pred_struct_counter).values()),
    )
    struct_metrics["required_field_coverage"] = (
        structure_matched / structure_required if structure_required else None
    )
    struct_metrics["required_fields_matched"] = structure_matched
    struct_metrics["required_fields_total"] = structure_required
    struct_metrics["missing_required"] = missing_required

    api_error_count = sum(len(item) if isinstance(item, list) else bool(item) for item in api_errors)
    api_status = "NOT_RUN" if not api_executed else "PASS" if api_error_count == 0 else "PARTIAL"
    score_values = [
        entity_metrics["f1"] if entity_gold else None,
        relation_metrics["f1"] if relation_gold else None,
        single_correct / single_total if single_total else None,
        prf(multi_tp, multi_fp, multi_fn)["f1"] if multi_counts else None,
        struct_metrics["required_field_coverage"] if structure_required else None,
    ]
    score_values = [value for value in score_values if value is not None]
    semantic_status = (
        "NOT_RUN" if not score_values else "PASS" if all(value == 1.0 for value in score_values) and not unsupported_relations else "PARTIAL"
    )
    primary_score = sum(score_values) / len(score_values) if score_values else None

    return {
        "api_validity": {"status": api_status, "error_count": api_error_count, "errors": api_errors},
        "semantic_quality": {
            "status": semantic_status,
            "primary_macro_score": primary_score,
            "entities_exact": entity_metrics,
            "relations_directional_exact": relation_metrics,
            "unsupported_custom_relation_instances": unsupported_relations,
            "single_label": {
                "correct": single_correct,
                "total": single_total,
                "accuracy": single_correct / single_total if single_total else None,
            },
            "multi_label": {
                "micro": prf(multi_tp, multi_fp, multi_fn),
                "macro_f1": sum(macro_values) / len(macro_values) if macro_values else None,
                "label_count": len(multi_counts),
            },
            "structures": struct_metrics,
        },
    }


def resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS requested but unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_model_spec(spec: str, index: int) -> tuple[str, str]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        if label and path:
            return label, path
    return f"model_{index}", spec


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    records = load_gold(args.data, args.max_examples)
    runs: dict[str, Any] = {}

    if args.predictions:
        predictions = load_predictions(args.predictions, len(records))
        runs["predictions"] = evaluate(
            records, predictions, [[] for _ in records], api_executed=False
        )
    else:
        import torch
        from gliner2 import GLiNER2

        device = resolve_device(torch, args.device)
        for index, spec in enumerate(args.model, 1):
            label, model_path = split_model_spec(spec, index)
            model = GLiNER2.from_pretrained(model_path, map_location=device)
            predictions: list[dict[str, Any]] = []
            errors: list[list[str]] = []
            for record in records:
                prediction, task_errors = predict_with_model(model, record, args.threshold)
                predictions.append(prediction)
                errors.append(task_errors)
            run = evaluate(records, predictions, errors, api_executed=True)
            run["runtime"] = {
                "model": model_path,
                "device": device,
                "parameter_devices": sorted({str(param.device) for param in model.parameters()}),
            }
            runs[label] = run

    labels = list(runs)
    comparison = None
    if len(labels) > 1:
        baseline = runs[labels[0]]["semantic_quality"]["primary_macro_score"]
        candidate = runs[labels[-1]]["semantic_quality"]["primary_macro_score"]
        comparison = {
            "baseline": labels[0],
            "candidate": labels[-1],
            "primary_macro_score_delta": (
                candidate - baseline if baseline is not None and candidate is not None else None
            ),
        }

    try:
        package_version = metadata.version("gliner2")
    except metadata.PackageNotFoundError:
        package_version = "source-checkout"
    report = {
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "gliner2_version": package_version,
        },
        "data": str(args.data.resolve()),
        "examples": len(records),
        "threshold": args.threshold,
        "runs": runs,
        "comparison": comparison,
    }
    rendered = json.dumps(json_safe(report), ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
