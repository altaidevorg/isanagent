#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"]
# ///
"""Safe full fine-tuning template for GLiNER2."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "fastino/gliner2-base-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate data, full-fine-tune GLiNER2, save metrics, and reload the final checkpoint."
    )
    parser.add_argument("--train-data", nargs="+", type=Path, help="Training JSONL path(s)")
    parser.add_argument("--eval-data", nargs="+", type=Path, help="Validation JSONL path(s)")
    parser.add_argument("--output-dir", type=Path, help="New or empty output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model ID or local model directory")
    parser.add_argument("--smoke-test", action="store_true", help="Use built-in data and exactly one optimizer step")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value overrides epoch-derived steps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--task-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--scheduler-type",
        choices=("linear", "cosine", "cosine_restarts", "constant"),
        default="linear",
    )
    parser.add_argument(
        "--eval-strategy", choices=("auto", "no", "steps", "epoch"), default="auto"
    )
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        help="Load full-model weights before training; optimizer/scheduler state is not resumed",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into a non-empty output directory; existing files are not removed",
    )
    parser.add_argument(
        "--allow-split-overlap",
        action="store_true",
        help="Allow identical normalized text in train and eval (normally treated as leakage)",
    )
    return parser.parse_args()


def smoke_examples() -> list[Any]:
    from gliner2.training.data import InputExample

    return [
        InputExample(
            text="Mira Chen works at Northwind Labs in Ankara.",
            entities={"person": ["Mira Chen"], "organization": ["Northwind Labs"], "location": ["Ankara"]},
        ),
        InputExample(
            text="Owen Park joined Contoso Research in Berlin.",
            entities={"person": ["Owen Park"], "organization": ["Contoso Research"], "location": ["Berlin"]},
        ),
    ]


def canonicalize_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
                if "input" in raw and "output" in raw:
                    records.append({"input": raw["input"], "output": raw["output"]})
                elif "text" in raw and "schema" in raw:
                    records.append({"input": raw["text"], "output": raw["schema"]})
                else:
                    raise ValueError(f"{path}:{line_no}: expected input/output or text/schema")
    return records


def validate_records(records: list[dict[str, Any]], label: str) -> tuple[Any, dict[str, Any]]:
    from gliner2.training.data import TrainingDataset

    if not records:
        raise ValueError(f"{label} dataset is empty")
    dataset = TrainingDataset.from_records(records)
    # Current GLiNER2 validation is always strict and has no strict= keyword.
    report = dataset.validate(raise_on_error=False)
    relation_errors = dataset.validate_relation_consistency()
    if report["invalid"] or relation_errors:
        details = report["errors"] + relation_errors
        raise ValueError(f"{label} dataset validation failed:\n- " + "\n- ".join(details[:20]))
    return dataset, {"validation": report, "relation_consistency_errors": relation_errors, "statistics": dataset.stats()}


def normalized_texts(dataset: Any) -> set[str]:
    return {" ".join(example.text.split()).casefold() for example in dataset}


def validate_cross_split_schemas(train_dataset: Any, eval_dataset: Any | None) -> None:
    if eval_dataset is None:
        return

    from gliner2.training.data import TrainingDataset

    combined = TrainingDataset([*train_dataset, *eval_dataset])
    relation_errors = combined.validate_relation_consistency()
    if relation_errors:
        raise ValueError(
            "Train/eval relation schemas are inconsistent:\n- " + "\n- ".join(relation_errors)
        )

    schemas: dict[str, tuple[tuple[str, ...], bool, str]] = {}
    errors: list[str] = []
    for split_name, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        for example_index, example in enumerate(dataset):
            for task in example.classifications:
                signature = (tuple(task.labels), bool(task.multi_label))
                previous = schemas.get(task.task)
                if previous is not None and signature != previous[:2]:
                    errors.append(
                        f"Classification task {task.task!r} changes labels/order or multi_label "
                        f"between {previous[2]} and {split_name}[{example_index}]"
                    )
                else:
                    schemas[task.task] = (*signature, f"{split_name}[{example_index}]")
    if errors:
        raise ValueError("Train/eval classification schemas are inconsistent:\n- " + "\n- ".join(errors))


def resolve_precision(torch: Any, requested: str) -> tuple[bool, bool, str]:
    if requested == "auto":
        requested = "fp16" if torch.cuda.is_available() else "fp32"
    if requested in {"fp16", "bf16"} and not torch.cuda.is_available():
        raise ValueError(f"{requested} requires CUDA with the current GLiNER2Trainer; use --precision fp32")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 was requested but torch.cuda.is_bf16_supported() is false")
    return requested == "fp16", requested == "bf16", requested


def command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        return subprocess.run(
            [executable, "--version"], check=False, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except OSError:
        return None


def environment_report(torch: Any) -> dict[str, Any]:
    try:
        gliner2_version = metadata.version("gliner2")
    except metadata.PackageNotFoundError:
        gliner2_version = "source-checkout"
    import gliner2

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "uv": command_version("uv"),
        "gliner2_version": gliner2_version,
        "gliner2_path": str(Path(gliner2.__file__).resolve()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def prepare_output(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        if not args.smoke_test:
            raise ValueError("--output-dir is required unless --smoke-test is used")
        return Path(tempfile.mkdtemp(prefix="gliner2-full-smoke-"))
    output = args.output_dir.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not args.allow_existing_output:
        raise ValueError(
            f"Output directory is not empty: {output}. Choose a new path or pass --allow-existing-output."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def reload_probe(model: Any, first_example: Any) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if first_example.entities:
        results["entities"] = model.extract_entities(
            first_example.text, list(first_example.entities), threshold=0.1
        )
    if first_example.classifications:
        tasks = {
            task.task: {
                "labels": task.labels,
                "multi_label": task.multi_label,
                **({"prompt": task.prompt} if task.prompt else {}),
                **({"examples": task.examples} if task.examples else {}),
                **(
                    {"label_descriptions": task.label_descriptions}
                    if task.label_descriptions
                    else {}
                ),
            }
            for task in first_example.classifications
        }
        results["classifications"] = model.classify_text(first_example.text, tasks, threshold=0.1)
    if first_example.structures:
        structures: dict[str, list[dict[str, Any]]] = {}
        for structure in first_example.structures:
            parent, values = next(iter(structure.to_dict().items()))
            fields = structures.setdefault(parent, [])
            known = {field["name"] for field in fields}
            for name, value in values.items():
                if name in known:
                    continue
                spec: dict[str, Any] = {
                    "name": name,
                    "dtype": "list" if isinstance(value, list) else "str",
                }
                if isinstance(value, dict) and "choices" in value:
                    spec["choices"] = value["choices"]
                fields.append(spec)
        results["structures"] = model.extract_json(first_example.text, structures, threshold=0.1)
    if first_example.relations:
        names = sorted({relation.name for relation in first_example.relations})
        results["relations"] = model.extract_relations(first_example.text, names, threshold=0.1)
    if not results:
        raise RuntimeError("Validated example unexpectedly contains no probeable task")
    return {"tasks": sorted(results), "results": results}


def main() -> int:
    args = parse_args()
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or a positive integer")
    if args.encoder_lr <= 0 or args.task_lr <= 0:
        raise ValueError("Learning rates must be positive")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("--warmup-ratio must be in [0, 1]")
    if args.eval_steps <= 0:
        raise ValueError("--eval-steps must be positive")
    if not args.smoke_test and not args.train_data:
        raise ValueError("--train-data is required unless --smoke-test is used")
    if args.smoke_test and (args.train_data or args.eval_data):
        raise ValueError("--smoke-test uses built-in bounded data; do not also pass data paths")
    if args.warm_start_checkpoint and (args.warm_start_checkpoint / "adapter_config.json").exists():
        raise ValueError("This full-fine-tuning script does not accept adapter-only checkpoints")

    from gliner2.training.data import TrainingDataset

    if args.smoke_test:
        train_dataset = TrainingDataset(smoke_examples())
        train_dataset, train_report = validate_records(train_dataset.to_records(), "train")
        eval_dataset = None
        eval_report = None
    else:
        train_dataset, train_report = validate_records(canonicalize_jsonl(args.train_data), "train")
        if args.eval_data:
            eval_dataset, eval_report = validate_records(canonicalize_jsonl(args.eval_data), "eval")
        else:
            eval_dataset, eval_report = None, None

    if eval_dataset is not None:
        validate_cross_split_schemas(train_dataset, eval_dataset)
        overlap = sorted(normalized_texts(train_dataset) & normalized_texts(eval_dataset))
        if overlap and not args.allow_split_overlap:
            raise ValueError(
                f"Detected {len(overlap)} normalized text(s) in both train and eval; split before training"
            )
    else:
        overlap = []

    output_dir = prepare_output(args)

    import torch
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    fp16, bf16, resolved_precision = resolve_precision(torch, args.precision)
    eval_strategy = args.eval_strategy
    if eval_strategy == "auto":
        eval_strategy = "epoch" if eval_dataset is not None else "no"
    if eval_strategy != "no" and eval_dataset is None:
        raise ValueError("Evaluation strategy requires --eval-data")
    if args.early_stopping and (eval_dataset is None or eval_strategy == "no"):
        raise ValueError("Early stopping requires eval data and an active eval strategy")

    max_steps = 1 if args.smoke_test else args.max_steps
    save_best = bool(args.save_best and eval_dataset is not None and eval_strategy != "no")
    config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name="gliner2-full-finetune",
        num_epochs=args.num_epochs,
        max_steps=max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        fp16=fp16,
        bf16=bf16,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        save_best=save_best,
        logging_steps=args.logging_steps,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        seed=args.seed,
        deterministic=args.deterministic,
        validate_data=False,
        max_len=args.max_len,
        use_lora=False,
        save_adapter_only=False,
    )

    environment = environment_report(torch)
    resolved = {
        "environment": environment,
        "requested_model": args.model,
        "output_dir": str(output_dir),
        "resolved_precision": resolved_precision,
        "config": asdict(config),
        "train_data": train_report,
        "eval_data": eval_report,
        "train_eval_overlap_count": len(overlap),
        "warm_start_checkpoint": str(args.warm_start_checkpoint.resolve()) if args.warm_start_checkpoint else None,
        "warm_start_resumes_optimizer": False,
    }
    print(json.dumps(json_safe(resolved), ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    # Data has been parsed, strictly validated, and checked for leakage before this model load.
    model = GLiNER2.from_pretrained(args.model)
    trainer = GLiNER2Trainer(model=model, config=config)
    if args.warm_start_checkpoint:
        trainer.load_checkpoint(str(args.warm_start_checkpoint.resolve()))
    resolved["trainer_device"] = str(trainer.device)
    resolved["model_parameter_devices"] = sorted({str(param.device) for param in trainer.model.parameters()})

    results = trainer.train(train_data=train_dataset, eval_data=eval_dataset)
    final_dir = output_dir / "final"
    if not final_dir.is_dir():
        raise RuntimeError(f"Trainer did not create expected final checkpoint: {final_dir}")

    reloaded = GLiNER2.from_pretrained(str(final_dir), map_location="cpu")
    probe = reload_probe(reloaded, train_dataset[0])
    summary = {
        **resolved,
        "training_results": results,
        "final_checkpoint": str(final_dir),
        "reload": {
            "status": "PASS",
            "class": type(reloaded).__name__,
            "parameter_devices": sorted({str(param.device) for param in reloaded.parameters()}),
            "probe": probe,
        },
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved machine-readable summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
