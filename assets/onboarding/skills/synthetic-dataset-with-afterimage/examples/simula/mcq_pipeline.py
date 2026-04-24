"""
MCQ datapoint with requirement critic + **double-critic** (verifiable label gate).

Paper mapping:
  - Taxonomy / mix / meta-prompt: same global → local pipeline as §2.2 and Algorithm 2.
  - Double-critic: two independent structured probes (“correct” vs “incorrect”) to
    reduce sycophancy on labeled answers (§2.2, §3.1, Fig. 3). Runs **after** the
    requirement critic loop accepts the JSON (OpenSimula implementation choice).

This example targets **four-option** items in the style of technical reading comprehension
(similar *format* to multiple-choice benchmarks discussed in the paper, e.g. CTI-MCQ /
Global MMLU), without reproducing any benchmark text.

Model: **gemini-2.5-flash** (paper-aligned cheap teacher; see blog / §3.2).

Requires: GEMINI_API_KEY

For checkpoints and batch workflows, see ``corpus_batch_qa.py`` (single-QA + JSONL) and ``README.md``.
"""

from __future__ import annotations

import asyncio
import os
import sys

from tqdm.auto import tqdm

from afterimage.providers import LLMFactory
from afterimage.simula import OpenSimula, configure_example_console

configure_example_console()

INSTRUCTION_Y = """\
Generate synthetic **four-option multiple-choice questions** for an internal
assessment on **software supply chain and incident readiness** (SBOM basics, severity
triaging, secure defaults). Each question must test reading-comprehension of concepts
that appear in typical engineering handbooks—not trivia about version numbers.
Distractors should be plausible misconceptions. One correct option only.\
"""

TARGET_DEPTH_D = 2
PROPOSAL_N = 3
OPEN_SIMULA_TEMPERATURE = 0.4
META_PROMPT_K = 6
COMPLEXIFY_C = 0.32
NUM_CHOICES = 4

MODEL_NAME = "gemini-2.5-flash"
MAX_FACTORS = 4
MAX_CHILDREN_PER_NODE = 8
MAX_FRONTIER_PER_DEPTH = 12


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.", file=sys.stderr)
        sys.exit(1)

    llm = LLMFactory.create(
        provider="gemini",
        model_name=MODEL_NAME,
        api_key=api_key,
    )
    sim = OpenSimula(llm, temperature=OPEN_SIMULA_TEMPERATURE)

    print("Building taxonomy (tqdm; httpx/google_genai muted)…\n", flush=True)
    bundle = await sim.build_taxonomy(
        INSTRUCTION_Y,
        document_provider=None,
        target_depth_D=TARGET_DEPTH_D,
        proposal_N=PROPOSAL_N,
        max_factors=MAX_FACTORS,
        max_children_per_node=MAX_CHILDREN_PER_NODE,
        max_frontier_per_depth=MAX_FRONTIER_PER_DEPTH,
        show_progress=True,
    )
    print()
    OpenSimula.validate_taxonomy_bundle(bundle)

    tail = tqdm(
        total=4,
        desc="OpenSimula │ after taxonomy",
        unit="step",
        dynamic_ncols=True,
    )
    tail.set_postfix_str("infer strategies")
    spec = await sim.infer_strategies(bundle)
    tail.update(1)
    tail.set_postfix_str("sample mix")
    mix = sim.sample_mix(bundle, spec)
    tail.update(1)
    tail.set_postfix_str("meta-prompts")
    meta = await sim.draw_meta_prompt(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        K=META_PROMPT_K,
        complexify_c=COMPLEXIFY_C,
        sequential=False,
    )
    tail.update(1)
    tail.set_postfix_str("MCQ + critics")
    row = await sim.generate_mcq_datapoint(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
        num_choices=NUM_CHOICES,
    )
    tail.update(1)
    tail.close()
    if row is None:
        print("No MCQ accepted (requirement loop and/or double-critic).")
    else:
        print(row.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
