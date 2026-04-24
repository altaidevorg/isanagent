"""Evaluate sample conversations with ConversationJudge (embeddings + LLM rubrics).

Requires GEMINI_API_KEY. Uses Gemini for both chat (judge LLM) and embeddings.

Run from the repository root::

    uv run examples/conversation_judge_demo.py

Or::

    python examples/conversation_judge_demo.py
"""

from __future__ import annotations

import asyncio
import os

from afterimage import (
    ConversationJudge,
    ConversationJudgeConfig,
    EmbeddingProviderFactory,
    LLMFactory,
    SmartKeyPool,
)
from afterimage.types import ConversationEntry, ConversationWithContext, Role


def _print_evaluated(title: str, row: ConversationWithContext) -> None:
    ev = row.evaluation
    assert ev is not None
    print(f"\n=== {title} ===")
    print(f"final_score: {row.final_score:.3f}")
    print(f"overall_grade: {ev.overall_grade.value}")
    for name in ("coherence", "grounding", "relevance", "factuality", "helpfulness"):
        entry = getattr(ev, name)
        fb = entry.feedback.strip()
        if len(fb) > 120:
            fb = fb[:117] + "..."
        print(f"  {name}: score={entry.score:.3f} | {fb}")


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.")
        return

    pool = SmartKeyPool.from_single_key(api_key)
    llm = LLMFactory.create(
        provider="gemini", model_name="gemini-2.0-flash", api_key=pool
    )
    embedding = EmbeddingProviderFactory.create(
        {"type": "gemini", "model": "gemini-embedding-001"},
        key_pool=pool,
    )

    config = ConversationJudgeConfig(
        min_acceptable_score=0.55,
        perfect_threshold=0.85,
        good_threshold=0.7,
        needs_improvement_threshold=0.5,
        bad_threshold=0.3,
    )
    judge = ConversationJudge(
        llm=llm,
        embedding_provider=embedding,
        config=config,
    )

    context = (
        "Riverdale Cafe opens daily at 7:00. We offer oat milk and soy milk. "
        "Pastries are baked in-house."
    )

    aligned = ConversationWithContext(
        conversations=[
            ConversationEntry(
                role=Role.USER,
                content="Do you have non-dairy milk?",
            ),
            ConversationEntry(
                role=Role.ASSISTANT,
                content="Yes — we offer oat milk and soy milk.",
            ),
        ],
        instruction_context=context,
        response_context=context,
    )

    weak = ConversationWithContext(
        conversations=[
            ConversationEntry(
                role=Role.USER,
                content="What are your hours?",
            ),
            ConversationEntry(
                role=Role.ASSISTANT,
                content="I am not sure; maybe check another city.",
            ),
        ],
        instruction_context=context,
        response_context=context,
    )

    try:
        out_ok = await judge.aevaluate_row(aligned)
        _print_evaluated("Aligned with context", out_ok)

        out_weak = await judge.aevaluate_row(weak)
        _print_evaluated("Weaker / less grounded", out_weak)
    finally:
        await judge.aclose()

    print(
        "\nTip: ConversationGenerator(..., auto_improve=True) builds a judge "
        "automatically; pass embedding_provider or embedding_provider_config "
        "to override defaults."
    )


if __name__ == "__main__":
    asyncio.run(main())
