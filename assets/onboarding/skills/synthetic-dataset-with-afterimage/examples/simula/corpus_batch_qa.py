"""
Multi-sample single-QA run with a **larger policy corpus**, **checkpoints**, and **JSONL export**.

This script is the place for “production-ish” ergonomics that we keep out of
``minimal_pipeline.py``: resume from disk, optional Hub push, concurrent sample
generation with incremental ``data/train.jsonl`` writes (crash-safe append).

Layout under ``--output-dir``::

    <output-dir>/
      opensimula/          # manifest + taxonomy + strategy + typed run_config (OpenSimulaRunConfig)
      data/
        train.jsonl        # one JSON object per line (accepted DataPointRecord only)

Requires: GEMINI_API_KEY. Optional: HF_TOKEN + ``--push-hf`` for upload after save.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path

from tqdm.auto import tqdm

from afterimage.providers import InMemoryDocumentProvider, LLMFactory
from afterimage.simula import (
    Checkpointer,
    OpenSimula,
    OpenSimulaRunConfig,
    append_datapoints_jsonl,
    configure_example_console,
    load_checkpoint,
)

configure_example_console()

INSTRUCTION_Y = """\
You are generating synthetic **security and compliance Q&A** for employees.
Ground every answer in the policy excerpts: cite concrete procedures, channels, or
classification rules that appear in the text. Do not invent statutes, vendors, or
tools not implied by the excerpts. Question ≤140 words; answer ≤200 words; neutral
professional tone.\
"""

# Slightly larger synthetic “corpus” (still static strings—swap for files if you like).
CORPUS_EXCERPTS = [
    """**Acceptable use.** Company devices and accounts may be monitored. Users must not
disable EDR, must use MFA for remote access, and must report suspected phishing within
one hour to the security mailbox. Customer PII must not be stored on personal cloud drives.\
""",
    """**Classification.** Restricted data includes credentials, live customer PII, and
unreleased financials. Restricted data must use approved encrypted channels only.
Managers must complete annual ransomware tabletops.\
""",
    """**Access lifecycle.** Contractors receive least-privilege roles; access is revoked
within 24 hours of offboarding. Shared mailboxes require documented owners and quarterly
access reviews.\
""",
    """**Incidents.** P1/P2 incidents page the SOC on-call immediately; P3/P4 follow the
next-business-day queue. Evidence preservation steps must not tip off suspected insiders.\
""",
    """**Third parties.** Vendors with access to customer data must sign the standard DPA
and provide SOC2 or equivalent annually. Exceptions require CISO approval with compensating
controls documented.\
""",
    """**Secure development.** Production secrets never live in git. CI must run SAST on
default branches; critical findings block release until waived by security engineering with
an expiry date.\
""",
]

MODEL_NAME = "gemini-2.5-flash"
OPEN_SIMULA_TEMPERATURE = 0.4
TARGET_DEPTH_D = 2
PROPOSAL_N = 3
META_PROMPT_K = 6
COMPLEXIFY_C = 0.28
MAX_FACTORS = 4
MAX_CHILDREN_PER_NODE = 8
MAX_FRONTIER_PER_DEPTH = 12


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/simula_corpus_batch"),
        help="Run directory (opensimula/ + data/train.jsonl).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of independent (mix, meta, generate) draws.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Load opensimula/ from output-dir; skip taxonomy and strategy inference.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Max concurrent sample pipelines (each does mix + meta + critic loop).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for mix/meta subsampling.")
    p.add_argument(
        "--push-hf",
        default=None,
        metavar="REPO_ID",
        help="After writing opensimula/, upload to this Hub dataset repo (needs HF_TOKEN).",
    )
    return p.parse_args()


async def main() -> None:
    args = _parse_args()
    out = args.output_dir.resolve()
    data_dir = out / "data"
    jsonl_path = data_dir / "train.jsonl"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.", file=sys.stderr)
        sys.exit(1)

    llm = LLMFactory.create(
        provider="gemini",
        model_name=MODEL_NAME,
        api_key=api_key,
    )
    docs = InMemoryDocumentProvider(CORPUS_EXCERPTS)
    sim = OpenSimula(llm, temperature=OPEN_SIMULA_TEMPERATURE)
    rng = random.Random(args.seed)

    if args.resume:
        print(f"Resume: loading checkpoint from {out / 'opensimula'}\n", flush=True)
        ckpt = load_checkpoint(out)
        bundle = ckpt.bundle
        spec = ckpt.sampling_strategy
        if spec is None:
            print("No sampling_strategy.json; inferring strategies…", flush=True)
            spec = await sim.infer_strategies(bundle)
    else:
        out.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        print("Building taxonomy (multi-document corpus)…\n", flush=True)
        bundle = await sim.build_taxonomy(
            INSTRUCTION_Y,
            document_provider=docs,
            target_depth_D=TARGET_DEPTH_D,
            proposal_N=PROPOSAL_N,
            max_factors=MAX_FACTORS,
            max_children_per_node=MAX_CHILDREN_PER_NODE,
            max_frontier_per_depth=MAX_FRONTIER_PER_DEPTH,
            show_progress=True,
        )
        print()
        OpenSimula.validate_taxonomy_bundle(bundle)
        spec = await sim.infer_strategies(bundle)

        run_cfg = OpenSimulaRunConfig(
            name="corpus_batch_qa",
            description="Multi-sample single-QA with policy corpus (examples/simula).",
            model=MODEL_NAME,
            temperature=OPEN_SIMULA_TEMPERATURE,
            target_depth_D=TARGET_DEPTH_D,
            proposal_N=PROPOSAL_N,
            meta_prompt_K=META_PROMPT_K,
            complexify_c=COMPLEXIFY_C,
            max_factors=MAX_FACTORS,
            max_children_per_node=MAX_CHILDREN_PER_NODE,
            max_frontier_per_depth=MAX_FRONTIER_PER_DEPTH,
            num_samples=args.num_samples,
            max_concurrency=args.max_concurrency,
            seed=args.seed,
            data_jsonl=str(jsonl_path.relative_to(out)),
            corpus_excerpt_count=len(CORPUS_EXCERPTS),
        )
        with Checkpointer(out) as cp:
            bundle.save(cp)
            spec.save(cp)
            cp.write_run_config(run_cfg)
        assert cp.manifest is not None
        print(
            f"Checkpoint written ({cp.manifest.format} {cp.manifest.format_version}) → {out / 'opensimula'}\n",
            flush=True,
        )
        if args.push_hf:
            url = cp.push_to_hub(args.push_hf)
            print(f"Pushed to Hub: {url}\n", flush=True)

    if args.num_samples <= 0:
        print("Nothing to generate (--num-samples <= 0).", flush=True)
        return

    accepted = 0
    pbar = tqdm(
        total=args.num_samples,
        desc="Generating samples",
        unit="sample",
        dynamic_ncols=True,
    )
    async for _idx, rec in sim.aiter_single_qa_samples(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        spec=spec,
        n=args.num_samples,
        K=META_PROMPT_K,
        complexify_c=COMPLEXIFY_C,
        sequential=False,
        max_concurrency=args.max_concurrency,
        rng=rng,
    ):
        if rec is not None:
            append_datapoints_jsonl(jsonl_path, [rec])
            accepted += 1
        pbar.set_postfix_str(f"accepted={accepted}")
        pbar.update(1)
    pbar.close()

    print(
        f"\nDone: {accepted}/{args.num_samples} accepted rows appended to {jsonl_path}",
        flush=True,
    )
    if not args.resume and args.push_hf is None:
        print(
            "Tip: re-run with --resume to skip taxonomy, or --push-hf org/repo after a save.",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
