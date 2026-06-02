#!/usr/bin/env python3
"""Aggregate compaction + reflection telemetry from a conversation.jsonl log.

Usage:
    python3 scripts/compaction_stats.py <path-to-conversation.jsonl>

Reads `BusMessage::Telemetry` entries written by `LoggingActor::write_conversation`
(see src/logging.rs:192) and prints aggregate stats for the Phase 0 compaction +
reflection variants:

- CompactionTriggered / CompactionCompleted / CompactionFailed
- ReflectionStarted / ReflectionCompleted

Reports counts, failure rate, median + p99 wall_ms, median compression ratio.
Skips other telemetry variants silently. Designed to run on any modern Python 3.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


VARIANT_KEYS = {
    "CompactionTriggered",
    "CompactionCompleted",
    "CompactionFailed",
    "ReflectionStarted",
    "ReflectionCompleted",
    # PR-6.1: AgentUsage carries cache_read_tokens / cache_creation_tokens so we
    # can compute the cache-hit ratio that PR-6 enables.
    "AgentUsage",
}


def iter_events(path: Path):
    """Yield (variant_name, payload_dict) for each compaction/reflection event in the file."""
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Conversation log mixes Inbound/Outbound/Telemetry; only Telemetry
                # variants emit as `{"VariantName": {...}}`. Other shapes are skipped.
                continue
            if not isinstance(obj, dict) or len(obj) != 1:
                continue
            variant, payload = next(iter(obj.items()))
            if variant in VARIANT_KEYS and isinstance(payload, dict):
                yield variant, payload


def percentile(values: list[float], q: float) -> float:
    """Plain percentile (linear interp) without numpy. `q` in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(__doc__ or "")
        return 2
    path = Path(argv[1])
    if not path.is_file():
        sys.stderr.write(f"error: {path} is not a file\n")
        return 1

    counts: Counter[str] = Counter()
    compaction_wall_ms: list[int] = []
    compaction_tokens_before: list[int] = []
    compaction_tokens_after: list[int] = []
    compaction_preprocess_ratios: list[float] = []
    compaction_failure_reasons: Counter[str] = Counter()
    reflection_wall_ms_short: list[int] = []
    reflection_wall_ms_long: list[int] = []
    # PR-6.1 cache accounting (across all `AgentUsage` events)
    cache_total_prompt = 0
    cache_total_read = 0
    cache_total_creation = 0

    for variant, payload in iter_events(path):
        counts[variant] += 1
        if variant == "AgentUsage":
            cache_total_prompt += int(payload.get("prompt_tokens", 0))
            cache_total_read += int(payload.get("cache_read_tokens", 0))
            cache_total_creation += int(payload.get("cache_creation_tokens", 0))
        if variant == "CompactionTriggered":
            # PR-1: track preprocess ratio (`tokens_after_preprocess / tokens_before`)
            # when present. Older logs predating PR-1 omit the field — `#[serde(default)]`
            # on the Rust side fills it as 0, so we skip the ratio computation in that case.
            tokens_before = int(payload.get("tokens_before", 0))
            after = int(payload.get("tokens_after_preprocess", 0))
            if tokens_before > 0 and after > 0:
                compaction_preprocess_ratios.append(after / tokens_before)
        elif variant == "CompactionCompleted":
            compaction_wall_ms.append(int(payload.get("wall_ms", 0)))
            compaction_tokens_before.append(int(payload.get("tokens_before", 0)))
            compaction_tokens_after.append(int(payload.get("tokens_after", 0)))
        elif variant == "CompactionFailed":
            compaction_failure_reasons[str(payload.get("reason", "unknown"))] += 1
        elif variant == "ReflectionCompleted":
            kind = payload.get("kind", "")
            if kind == "ShortTerm":
                reflection_wall_ms_short.append(int(payload.get("wall_ms", 0)))
            elif kind == "LongTerm":
                reflection_wall_ms_long.append(int(payload.get("wall_ms", 0)))

    print(f"== Compaction telemetry — {path} ==")
    print()
    print("Event counts:")
    for variant in sorted(VARIANT_KEYS):
        print(f"  {variant:<25} {counts[variant]}")
    print()

    triggered = counts["CompactionTriggered"]
    completed = counts["CompactionCompleted"]
    failed = counts["CompactionFailed"]
    accounted = completed + failed
    if triggered:
        failure_rate = failed / triggered if triggered else 0.0
        print(f"Compaction failure rate: {failure_rate:.1%} ({failed}/{triggered})")
        orphan = triggered - accounted
        if orphan:
            print(
                f"  WARN: {orphan} CompactionTriggered without matching Completed/Failed "
                "(see Phase 0 acceptance criteria)"
            )

    if compaction_wall_ms:
        p50 = percentile([float(x) for x in compaction_wall_ms], 50)
        p99 = percentile([float(x) for x in compaction_wall_ms], 99)
        print(f"Compaction wall_ms: p50={p50:.0f}ms  p99={p99:.0f}ms  n={len(compaction_wall_ms)}")

    if compaction_tokens_before and compaction_tokens_after:
        ratios = [
            (a / b) if b else 0.0
            for a, b in zip(compaction_tokens_after, compaction_tokens_before)
        ]
        median_ratio = statistics.median(ratios)
        median_before = statistics.median(compaction_tokens_before)
        median_after = statistics.median(compaction_tokens_after)
        print(
            f"Compaction compression: median tokens {median_before:.0f} → {median_after:.0f} "
            f"(ratio {median_ratio:.3f})"
        )

    if compaction_preprocess_ratios:
        # PR-1 acceptance criterion target: ≥30% reduction on image-/tool-heavy
        # workloads, i.e. preprocess_ratio ≤ 0.70. Lower is better.
        median_pp = statistics.median(compaction_preprocess_ratios)
        print(
            f"Compaction preprocess ratio (after/before): median={median_pp:.3f}  "
            f"n={len(compaction_preprocess_ratios)}"
        )

    if cache_total_prompt > 0:
        # PR-6.1 cache effectiveness across all AgentUsage events. A high cache_read
        # ratio means PR-6's system-prompt caching is hitting; cache_creation only
        # spikes on cold sessions. OpenAI providers leave cache_creation at 0 (no
        # separate billing), so the ratio is most meaningful for Anthropic traffic.
        read_ratio = cache_total_read / cache_total_prompt
        create_ratio = cache_total_creation / cache_total_prompt
        print(
            f"\nProvider prompt-cache (all AgentUsage events, n={counts['AgentUsage']}):"
        )
        print(
            f"  prompt_tokens={cache_total_prompt}  "
            f"cache_read={cache_total_read} ({read_ratio:.1%})  "
            f"cache_create={cache_total_creation} ({create_ratio:.1%})"
        )

    if compaction_failure_reasons:
        print()
        print("Failure reasons:")
        for reason, n in compaction_failure_reasons.most_common():
            print(f"  {n:>4}  {reason}")

    if reflection_wall_ms_short:
        p50 = percentile([float(x) for x in reflection_wall_ms_short], 50)
        p99 = percentile([float(x) for x in reflection_wall_ms_short], 99)
        print(
            f"\nShort-term reflection wall_ms: p50={p50:.0f}ms  p99={p99:.0f}ms  "
            f"n={len(reflection_wall_ms_short)}"
        )
    if reflection_wall_ms_long:
        p50 = percentile([float(x) for x in reflection_wall_ms_long], 50)
        p99 = percentile([float(x) for x in reflection_wall_ms_long], 99)
        print(
            f"Long-term reflection wall_ms:  p50={p50:.0f}ms  p99={p99:.0f}ms  "
            f"n={len(reflection_wall_ms_long)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
