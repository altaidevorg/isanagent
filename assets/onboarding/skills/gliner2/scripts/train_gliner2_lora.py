# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c",
# ]
# ///
"""Train and round-trip a PEFT-native GLiNER2 LoRA adapter.

Run with ``uv run --script train_gliner2_lora.py --help``. A normal run
requires explicit training data. ``--smoke-test`` is the only mode that uses
bundled synthetic examples and always limits optimization to one step.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


DEFAULT_MODEL = "fastino/gliner2-base-v1"
SUPPORTED_TARGETS = (
    "encoder",
    "encoder.query",
    "encoder.key",
    "encoder.value",
    "encoder.dense",
    "span_rep",
    "classifier",
    "count_embed",
    "count_pred",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate GLiNER2 data, train a PEFT-native LoRA adapter, and "
            "verify a fresh-base adapter round trip."
        )
    )
    parser.add_argument("--train-data", type=Path, nargs="+", help="Training JSONL path(s).")
    parser.add_argument("--eval-data", type=Path, nargs="+", help="Optional evaluation JSONL path(s).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gliner2-lora"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model ID or verified local snapshot.")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic data and force max_steps=1.")
    parser.add_argument("--validate-only", action="store_true", help="Validate data/config without loading a model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--task-lr", type=float, default=5e-4, help="Learning rate for LoRA parameters.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--use-dora", action="store_true")
    parser.add_argument(
        "--lora-target",
        action="append",
        choices=SUPPORTED_TARGETS,
        help="Repeat to select target groups; defaults to encoder.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
        help="Preflight expectation. The current trainer itself auto-selects CUDA, otherwise CPU.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument(
        "--merge-output",
        type=Path,
        help="Optionally save a separately merged full model after round-trip verification.",
    )
    return parser


def positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke_test and args.train_data:
        raise ValueError("--smoke-test supplies synthetic data; do not also pass --train-data")
    if args.smoke_test and args.eval_data:
        raise ValueError("--smoke-test supplies synthetic data; do not also pass --eval-data")
    if not args.smoke_test and not args.train_data:
        raise ValueError("--train-data is required unless --smoke-test is used")
    positive("epochs", args.epochs)
    positive("batch-size", args.batch_size)
    positive("eval-batch-size", args.eval_batch_size)
    positive("gradient-accumulation-steps", args.gradient_accumulation_steps)
    positive("task-lr", args.task_lr)
    positive("logging-steps", args.logging_steps)
    positive("eval-steps", args.eval_steps)
    positive("lora-r", args.lora_r)
    positive("lora-alpha", args.lora_alpha)
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or a positive integer")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("--warmup-ratio must be in [0, 1]")
    if args.output_dir.exists() and not args.validate_only:
        raise ValueError(f"Output directory already exists; choose a new path: {args.output_dir}")
    if args.merge_output and args.merge_output.exists():
        raise ValueError(f"Merged output path already exists; choose a new path: {args.merge_output}")


def make_smoke_datasets() -> tuple[Any, Any]:
    from gliner2.training.data import InputExample, TrainingDataset

    train = TrainingDataset(
        [
            InputExample(
                text="Northwind Supplies invoiced Contoso Retail for USD 210.",
                entities={"company": ["Northwind Supplies", "Contoso Retail"], "amount": ["USD 210"]},
            ),
            InputExample(
                text="Fabrikam paid Adventure Works EUR 90 on 2026-07-10.",
                entities={
                    "company": ["Fabrikam", "Adventure Works"],
                    "amount": ["EUR 90"],
                    "date": ["2026-07-10"],
                },
            ),
        ]
    )
    evaluation = TrainingDataset(
        [
            InputExample(
                text="Contoso Retail paid Northwind Supplies USD 75.",
                entities={"company": ["Contoso Retail", "Northwind Supplies"], "amount": ["USD 75"]},
            )
        ]
    )
    return train, evaluation


def load_and_validate(paths: Sequence[Path], label: str) -> Any:
    from gliner2.training.data import TrainingDataset

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label} data: {missing}")
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"{path}:{line_no}: top-level JSON value must be an object")
                if "input" in raw and "output" in raw:
                    records.append({"input": raw["input"], "output": raw["output"]})
                elif "text" in raw and "schema" in raw:
                    records.append({"input": raw["text"], "output": raw["schema"]})
                else:
                    raise ValueError(f"{path}:{line_no}: expected input/output or text/schema")
    dataset = TrainingDataset.from_records(records)
    if len(dataset) == 0:
        raise ValueError(f"{label} dataset is empty")
    report = dataset.validate(raise_on_error=True)
    relation_errors = dataset.validate_relation_consistency()
    if relation_errors:
        raise ValueError(f"{label} relation schema is inconsistent:\n" + "\n".join(relation_errors))
    print(json.dumps({f"{label}_validation": report}, indent=2, default=str))
    return dataset


def validate_cross_split_consistency(train_data: Any, eval_data: Any | None) -> None:
    if eval_data is None:
        return

    from gliner2.training.data import TrainingDataset

    train_texts = {" ".join(example.text.split()).casefold() for example in train_data}
    eval_texts = {" ".join(example.text.split()).casefold() for example in eval_data}
    overlap = train_texts & eval_texts
    if overlap:
        raise ValueError(
            f"Detected {len(overlap)} normalized text(s) in both train and eval; split before training"
        )

    combined = TrainingDataset([*train_data, *eval_data])
    relation_errors = combined.validate_relation_consistency()
    if relation_errors:
        raise ValueError(
            "Train/eval relation schemas are inconsistent:\n" + "\n".join(relation_errors)
        )

    classification_schemas: dict[str, tuple[tuple[str, ...], bool, str]] = {}
    classification_errors: list[str] = []
    for split_name, dataset in (("train", train_data), ("eval", eval_data)):
        for example_index, example in enumerate(dataset):
            for task in example.classifications:
                signature = (tuple(task.labels), bool(task.multi_label))
                previous = classification_schemas.get(task.task)
                if previous is not None and signature != previous[:2]:
                    classification_errors.append(
                        f"Classification task {task.task!r} changes labels/order or multi_label "
                        f"between {previous[2]} and {split_name}[{example_index}]"
                    )
                else:
                    classification_schemas[task.task] = (*signature, f"{split_name}[{example_index}]")
    if classification_errors:
        raise ValueError("Train/eval classification schemas are inconsistent:\n" + "\n".join(classification_errors))


def validate_dataset(dataset: Any, label: str) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{label} dataset is empty")
    report = dataset.validate(raise_on_error=True)
    relation_errors = dataset.validate_relation_consistency()
    if relation_errors:
        raise ValueError(f"{label} relation schema is inconsistent:\n" + "\n".join(relation_errors))
    print(json.dumps({f"{label}_validation": report}, indent=2, default=str))


def resolve_runtime(args: argparse.Namespace, torch: Any) -> tuple[str, str, bool, bool]:
    trainer_device = "cuda" if torch.cuda.is_available() else "cpu"
    requested = trainer_device if args.device == "auto" else args.device
    if requested == "mps":
        available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        suffix = " is available" if available else " is not available"
        raise RuntimeError(
            "The current GLiNER2Trainer does not select MPS (MPS" + suffix +
            "); choose CPU/CUDA for training."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested != trainer_device:
        raise RuntimeError(
            f"Requested {requested}, but current GLiNER2Trainer will select {trainer_device}. "
            "Make the requested device visible before launching the process."
        )

    precision = args.precision
    if precision == "auto":
        precision = "fp16" if trainer_device == "cuda" else "fp32"
    if precision in {"fp16", "bf16"} and trainer_device != "cuda":
        raise RuntimeError(f"{precision} training requires supported CUDA in this template")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but torch.cuda.is_bf16_supported() is false")
    return trainer_device, precision, precision == "fp16", precision == "bf16"


def adapter_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def verify_adapter_round_trip(trained: Any, loaded: Any) -> dict[str, Any]:
    import torch

    before = adapter_state(trained)
    after = adapter_state(loaded)
    if not before:
        raise AssertionError("The trained model contains no LoRA parameters")
    if before.keys() != after.keys():
        missing = sorted(before.keys() - after.keys())
        extra = sorted(after.keys() - before.keys())
        raise AssertionError(f"Adapter parameter names differ; missing={missing}, extra={extra}")
    for name in before:
        if before[name].shape != after[name].shape:
            raise AssertionError(f"Adapter shape differs for {name}")
        if before[name].dtype != after[name].dtype:
            raise AssertionError(f"Adapter dtype differs for {name}")
        if not torch.equal(before[name], after[name]):
            raise AssertionError(f"Adapter value differs for {name}")
    return {"adapter_parameter_tensors": len(before), "round_trip_equal": True}


def checkpoint_metadata(adapter_dir: Path, expected_base: str) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights = [adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin"]
    if not config_path.is_file() or not any(path.is_file() for path in weights):
        raise FileNotFoundError(f"Incomplete PEFT adapter checkpoint: {adapter_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("adapter_config.json is not a PEFT-native LORA config")
    base_id = config.get("base_model_name_or_path")
    if not base_id:
        raise ValueError("Adapter config has no base_model_name_or_path")
    if base_id != expected_base:
        raise ValueError(f"Adapter base mismatch: expected {expected_base!r}, found {base_id!r}")
    return {
        "adapter_dir": str(adapter_dir.resolve()),
        "base_model_name_or_path": base_id,
        "peft_type": config["peft_type"],
        "weight_file": next(path.name for path in weights if path.is_file()),
    }


def run_reload_probe(model: Any) -> dict[str, Any]:
    text = "Northwind Supplies invoiced Contoso Retail for USD 75."
    result = model.extract_entities(
        text,
        ["company", "amount"],
        include_confidence=True,
        include_spans=True,
    )
    entities = result.get("entities")
    if not isinstance(entities, dict):
        raise AssertionError("Freshly reloaded adapter returned an invalid entity output shape")
    span_count = 0
    for items in entities.values():
        for item in items:
            if not isinstance(item, dict) or not {"text", "start", "end"} <= item.keys():
                raise AssertionError("Span-bearing reload probe returned an invalid entity item")
            if text[item["start"]:item["end"]] != item["text"]:
                raise AssertionError(f"Invalid reload-probe span: {item}")
            span_count += 1
    return {"inference_succeeded": True, "valid_span_items": span_count}


def run_fresh_process_probe(adapter_dir: Path, base_model: str, device: str) -> dict[str, Any]:
    probe_code = r'''
import json
import sys
from gliner2 import GLiNER2
from peft import PeftModel

base_model, adapter_dir, device = sys.argv[1:]
text = "Northwind Supplies invoiced Contoso Retail for USD 75."
base = GLiNER2.from_pretrained(base_model, map_location=device)
adapted = PeftModel.from_pretrained(base, adapter_dir)
adapted.eval()
result = adapted.extract_entities(
    text,
    ["company", "amount"],
    include_confidence=True,
    include_spans=True,
)
entities = result.get("entities")
if not isinstance(entities, dict):
    raise AssertionError("Fresh-process adapter probe returned an invalid output shape")
span_count = 0
for items in entities.values():
    for item in items:
        if not isinstance(item, dict) or not {"text", "start", "end"} <= item.keys():
            raise AssertionError("Fresh-process adapter probe returned an invalid span item")
        if text[item["start"]:item["end"]] != item["text"]:
            raise AssertionError(f"Fresh-process adapter probe returned an invalid span: {item}")
        span_count += 1
print("GLINER2_PROBE_JSON=" + json.dumps({
    "status": "PASS",
    "python_executable": sys.executable,
    "valid_span_items": span_count,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe_code, base_model, str(adapter_dir), device],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Fresh-process adapter probe failed:\n"
            + (completed.stderr or completed.stdout)[-4000:]
        )
    marker = "GLINER2_PROBE_JSON="
    payload = next(
        (line[len(marker):] for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if payload is None:
        raise RuntimeError("Fresh-process adapter probe did not emit its JSON result")
    return json.loads(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if args.smoke_test:
        train_data, eval_data = make_smoke_datasets()
        validate_dataset(train_data, "train")
        validate_dataset(eval_data, "eval")
        max_steps = 1
    else:
        train_data = load_and_validate(args.train_data, "train")
        eval_data = load_and_validate(args.eval_data, "eval") if args.eval_data else None
        max_steps = args.max_steps

    validate_cross_split_consistency(train_data, eval_data)

    import torch

    device, precision, fp16, bf16 = resolve_runtime(args, torch)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "model_loaded": False,
                    "resolved_device": device,
                    "resolved_precision": precision,
                    "max_steps": max_steps,
                },
                indent=2,
            )
        )
        return 0

    import gliner2
    import peft
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
    from peft import PeftModel

    args.output_dir.mkdir(parents=True, exist_ok=False)
    targets = args.lora_target or ["encoder"]
    has_eval = eval_data is not None

    runtime = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "gliner2_version": gliner2.__version__,
        "gliner2_path": str(Path(gliner2.__file__).resolve()),
        "peft_version": peft.__version__,
        "torch_version": torch.__version__,
        "base_model": args.model,
        "device": device,
        "precision": precision,
    }
    print(json.dumps({"runtime": runtime}, indent=2))

    config = TrainingConfig(
        output_dir=str(args.output_dir),
        experiment_name=args.output_dir.name,
        num_epochs=args.epochs,
        max_steps=max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        task_lr=args.task_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=fp16,
        bf16=bf16,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=1 if args.smoke_test and has_eval else args.eval_steps,
        save_best=has_eval,
        early_stopping=False,
        logging_steps=args.logging_steps,
        report_to_wandb=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        seed=args.seed,
        deterministic=True,
        # Data was already strictly validated above; do not invoke the trainer's
        # sanitizing loader, which can silently drop invalid annotations.
        validate_data=False,
        max_len=args.max_len,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_use_dora=args.use_dora,
        lora_target_modules=targets,
        save_adapter_only=True,
    )

    base = GLiNER2.from_pretrained(args.model, map_location=device)
    trainer = GLiNER2Trainer(model=base, config=config)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    trainable_names = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
    if not trainable_names or not all("lora_" in name for name in trainable_names):
        raise AssertionError("The trainable set is empty or includes non-LoRA parameters")
    parameter_report = {
        "trainable": trainable,
        "total": total,
        "percentage": 100.0 * trainable / total,
        "target_groups": targets,
    }
    print(json.dumps({"parameters": parameter_report}, indent=2))

    results = trainer.train(train_data=train_data, eval_data=eval_data)
    adapter_dir = args.output_dir / "final"
    artifact = checkpoint_metadata(adapter_dir, args.model)

    fresh_base = GLiNER2.from_pretrained(args.model, map_location=device)
    reloaded = PeftModel.from_pretrained(fresh_base, str(adapter_dir))
    reloaded.eval()
    round_trip = verify_adapter_round_trip(trainer.model, reloaded)
    round_trip.update(run_reload_probe(reloaded))
    actual_devices = sorted({str(parameter.device) for parameter in reloaded.parameters()})
    actual_device_types = sorted({parameter.device.type for parameter in reloaded.parameters()})
    if actual_device_types != [device]:
        raise AssertionError(f"Reloaded model devices differ from expectation: {actual_devices}")

    merged_path = None
    if args.merge_output:
        args.merge_output.mkdir(parents=True, exist_ok=False)
        merged = reloaded.merge_and_unload()
        merged.save_pretrained(str(args.merge_output))
        GLiNER2.from_pretrained(str(args.merge_output), map_location=device)
        merged_path = str(args.merge_output.resolve())

    # Release in-process model graphs before proving that the adapter also loads
    # and infers correctly in a clean interpreter.
    del reloaded, fresh_base, trainer, base
    gc.collect()
    fresh_process = run_fresh_process_probe(adapter_dir, args.model, device)

    report = {
        "runtime": runtime,
        "config": asdict(config),
        "parameters": parameter_report,
        "training": results,
        "artifact": artifact,
        "round_trip": round_trip,
        "fresh_process_probe": fresh_process,
        "actual_devices": actual_devices,
        "actual_device_types": actual_device_types,
        "merged_model": merged_path,
    }
    report_path = args.output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(report_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
