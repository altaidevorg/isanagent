"""
Single-QA OpenSimula pipeline (requirement critic + refine; no double-critic).

Paper mapping (Davidson et al., TMLR; Appendix B.4, §2.2, Algorithm 2):
  - instruction_y  →  **y**: dataset specification.
  - document text  →  optional **S**: domain grounding (§2.1); capped when passed via DocumentProvider.
  - target_depth_D →  **D** (taxonomy depth); deeper = finer global coverage, higher cost.
  - proposal_N     →  **N** in Best-of-N child proposals before the critic merges them.
  - infer_strategies / sample_mix → joint sampling strategies and one **mix** (§2.2).
  - K              →  number of **meta-prompt** candidates; one is randomly kept (local diversity).
  - complexify_c   →  probability **c** of complexifying that meta-prompt (§2.2). Table 1 uses **c=0.5**
    for the paper’s “Local” system; we use a lower default here to limit difficulty skew while iterating.

Model: **gemini-2.5-flash** — same family as the paper’s teacher (Gemini 2.5 Flash), cheap for loops.

Requires: GEMINI_API_KEY

For checkpoints, multi-sample JSONL, and Hub upload, see ``corpus_batch_qa.py`` and ``examples/simula/README.md``.
"""

from __future__ import annotations

import asyncio
import os
import sys

from tqdm.auto import tqdm

from afterimage.providers import InMemoryDocumentProvider, LLMFactory
from afterimage.simula import OpenSimula, configure_example_console

configure_example_console()

INSTRUCTION_Y = """\
You are generating synthetic **training Q&A** for enterprise employees (security
and acceptable-use awareness). Each item must be grounded in the provided policy
excerpts: answers should cite concrete controls or procedures implied by the text,
not invent vendor-specific products or laws not mentioned. Target length: question
≤120 words, answer ≤180 words, factual tone, no panic language.\
"""

POLICY_EXCERPTS = [
    """\
**Corporate Acceptable Use (excerpt).** Company systems may be monitored to ensure
compliance. Users must not disable endpoint protection, must report suspected phishing
within one hour via the security mailbox, and must not store customer personal data
on unapproved cloud drives. Remote access requires MFA on every session. Contractors
receive least-privilege accounts revoked within 24 hours of offboarding.\
""",
    """\
**Data classification (excerpt).** "Restricted" data includes credentials, live
customer PII, and unreleased financials. Restricted data may only transit over
approved encrypted channels. Incident severity P1/P2 requires paging the on-call SOC;
P3/P4 is next-business-day. Tabletop exercises for ransomware are mandatory annually
for all people managers.\
""",
]

TARGET_DEPTH_D = 2
PROPOSAL_N = 3
OPEN_SIMULA_TEMPERATURE = 0.4
META_PROMPT_K = 6
COMPLEXIFY_C = 0.28

MODEL_NAME = "gemini-2.5-flash"
MAX_FACTORS = 4
MAX_CHILDREN_PER_NODE = 8
MAX_FRONTIER_PER_DEPTH = 12


async def main() -> None:
    print("Starting…", flush=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.", file=sys.stderr)
        sys.exit(1)

    llm = LLMFactory.create(
        provider="gemini",
        model_name=MODEL_NAME,
        api_key=api_key,
    )
    docs = InMemoryDocumentProvider(POLICY_EXCERPTS)
    sim = OpenSimula(llm, temperature=OPEN_SIMULA_TEMPERATURE)
    print("OpenSimula ready — taxonomy uses tqdm; httpx/google_genai logs are muted.\n", flush=True)

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

    tail = tqdm(
        total=4,
        desc="OpenSimula │ after taxonomy",
        unit="step",
        dynamic_ncols=True,
    )
    tail.set_postfix_str("infer strategies (§2.2)")
    spec = await sim.infer_strategies(bundle)
    tail.update(1)
    tail.set_postfix_str("sample mix")
    mix = sim.sample_mix(bundle, spec)
    tail.update(1)
    tail.set_postfix_str(f"meta-prompts (K={META_PROMPT_K})")
    meta = await sim.draw_meta_prompt(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        K=META_PROMPT_K,
        complexify_c=COMPLEXIFY_C,
        sequential=False,
    )
    tail.update(1)
    tail.set_postfix_str("single QA + critic")
    row = await sim.generate_single_qa_datapoint(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
    )
    tail.update(1)
    tail.close()

    if row is None:
        print("No row accepted (requirement critic or refine loop).", flush=True)
    else:
        print(row.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
