#!/usr/bin/env python3
"""Summarize evaluation samples into sample_summary.md.

Adapted from AutoTrainess (MIT) — https://github.com/simple-agent-lab/AutoTrainess

Supports:
- inspect_ai-style JSON/JSONL logs with sample fields
- generic JSONL with keys: score/input/target/output (flexible aliases)

Usage:
  python summarize_eval_samples.py --input path/to/eval_log.jsonl --output sample_summary.md [--n 15]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ALIASES = {
    "score": ("score", "metric", "accuracy", "correct", "reward"),
    "input": ("input", "prompt", "question", "messages", "user"),
    "target": ("target", "answer", "label", "gold", "reference"),
    "output": ("output", "completion", "prediction", "response", "model_output"),
}


def _pick(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _normalize(sample: dict[str, Any]) -> dict[str, Any] | None:
    # inspect_ai-ish nested score
    score = _pick(sample, ALIASES["score"])
    if score is None and isinstance(sample.get("scores"), dict):
        vals = list(sample["scores"].values())
        if vals:
            first = vals[0]
            if isinstance(first, dict):
                score = first.get("value", first.get("score"))
            else:
                score = first
    inp = _pick(sample, ALIASES["input"])
    target = _pick(sample, ALIASES["target"])
    out = _pick(sample, ALIASES["output"])
    if inp is None and out is None and target is None:
        return None
    return {"score": score, "input": inp, "target": target, "output": out}


def _load_samples(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    samples: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("samples") or data.get("results") or data.get("rows") or [data]
        else:
            raw = []
        for item in raw:
            if isinstance(item, dict):
                norm = _normalize(item)
                if norm:
                    samples.append(norm)
        return samples
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            norm = _normalize(obj)
            if norm:
                samples.append(norm)
    return samples


def _fmt(v: Any, limit: int = 800) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False, indent=2)
    else:
        s = str(v)
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples = _load_samples(args.input)
    if not samples:
        args.output.write_text(
            "# Sample summary\n\nNo parseable samples found in input.\n",
            encoding="utf-8",
        )
        return 1

    rng = random.Random(args.seed)
    n = min(args.n, len(samples))
    chosen = rng.sample(samples, n)

    parts = [
        "# Sample summary\n",
        f"Selected {n} of {len(samples)} samples (seed={args.seed}).\n",
    ]
    for i, s in enumerate(chosen, 1):
        parts.append(f"## Sample {i}\n")
        parts.append(f"- Score: {_fmt(s.get('score'), 120)}\n")
        parts.append(f"- Input:\n\n```\n{_fmt(s.get('input'))}\n```\n")
        parts.append(f"- Target:\n\n```\n{_fmt(s.get('target'))}\n```\n")
        parts.append(f"- Model output:\n\n```\n{_fmt(s.get('output'))}\n```\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {args.output} ({n} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
