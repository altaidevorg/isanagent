"""Generate synthetic caselaw-grounded dialogs (Qdrant RAG + Gemini).

Run from the **repository root** after ``uv sync`` (needs ``afterimage`` and
``embeddings-local`` for the default query embedding model)::

    uv run python examples/caselaw_rag/generate.py --help

Query vectors use **BAAI/bge-base-en-v1.5** (768-d) to match
``index_corpus.py`` on ``free-law/Caselaw_Access_Project_embeddings``.

See ``README.md`` in this directory for setup and Docker Qdrant steps.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import timedelta
from pathlib import Path

from afterimage import (
    ConversationGenerator,
    ContextualInstructionGeneratorCallback,
    EmbeddingProviderFactory,
    GenerationMonitor,
    WithRAGRespondentPromptModifier,
)
from afterimage.providers import QdrantDocumentProvider
from afterimage.retrievers import QdrantRetriever
from afterimage.storage import JSONLStorage
from qdrant_client import AsyncQdrantClient, QdrantClient

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output" / "conversations.jsonl"

RESPONDENT_PROMPT = """You are a careful senior legal research assistant helping
produce synthetic training dialogues. Ground every substantive claim in the
**retrieved court opinions and excerpts** supplied in your context. When the
excerpts include neutral identifiers (court, docket or neutral citation, date),
use them faithfully in your explanation. Use plain English and define legal jargon
when it helps a lay reader. If the retrieved material is insufficient to answer,
say what is missing instead of inventing holdings or citations. This is educational
synthetic data only; you are not providing real-world legal advice."""

CORRESPONDENT_PROMPT = """You are an experienced lawyer or a legally curious
client in a role-play. Your partner is a research assistant who answers from
retrieved case excerpts. Ask realistic questions and follow-ups about the legal
issues suggested by the case material you are given (for example contracts,
torts, criminal procedure, civil procedure, or administrative law). Stay in
character; do not break the fourth wall. For your first message, wait until you
are prompted to begin. Do not fabricate specific docket numbers or citations
unless they already appear in your briefing; otherwise stay at the level of
issues and facts inspired by the materials."""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL (env: QDRANT_URL)",
    )
    p.add_argument(
        "--qdrant-api-key",
        default=os.environ.get("QDRANT_API_KEY"),
        help="Optional Qdrant API key (env: QDRANT_API_KEY)",
    )
    p.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "caselaw_cap_demo"),
        help="Indexed collection name (env: QDRANT_COLLECTION)",
    )
    p.add_argument(
        "--content-key",
        default=os.environ.get("QDRANT_CONTENT_KEY", "content"),
        help="Payload field with opinion text (must match index_corpus.py)",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=int(os.environ.get("QDRANT_MAX_DOCS", "50")),
        help="Max documents QdrantDocumentProvider samples for instruction context",
    )
    p.add_argument(
        "--num-dialogs",
        type=int,
        default=int(os.environ.get("NUM_DIALOGS", "5")),
        help="Conversations to generate",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("MAX_TURNS", "1")),
        help="Max turns per dialog (uniform random 1..max_turns)",
    )
    p.add_argument(
        "--gemini-model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        help="Gemini model id",
    )
    p.add_argument(
        "--embedding-model",
        default=os.environ.get(
            "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
        ),
        help="SentenceTransformer id for query vectors (must match indexed vectors)",
    )
    p.add_argument(
        "--embedding-workers",
        type=int,
        default=int(os.environ.get("EMBEDDING_WORKERS", "2")),
        help="Process pool workers for local embeddings",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("CASELAW_OUTPUT", str(DEFAULT_OUT))),
        help="JSONL output path for conversations",
    )
    p.add_argument(
        "--auto-improve",
        action="store_true",
        help="Enable quality gate retries (needs embeddings-local for judge)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=HERE / "logs",
        help="GenerationMonitor log directory",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=HERE / "plots",
        help="Directory for monitor.plot outputs",
    )
    return p


def _qdrant_kwargs(url: str, api_key: str | None) -> dict:
    kwargs: dict = {"url": url, "timeout": 120.0}
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


async def _async_main(args: argparse.Namespace) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY for Gemini.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    def on_alert(alert) -> None:
        print(f"alert - {alert.name} - {alert.message}")

    monitor = GenerationMonitor(
        log_dir=str(args.log_dir),
        alert_handlers=[on_alert],
        metrics_interval=60,
    )

    qd_kw = _qdrant_kwargs(args.qdrant_url, args.qdrant_api_key)
    qd = QdrantClient(**qd_kw)
    qd_async = AsyncQdrantClient(**qd_kw)
    documents = QdrantDocumentProvider(
        client=qd,
        collection_name=args.collection,
        content_key=args.content_key,
        max_docs=args.max_docs,
    )

    instruction_cb = ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=documents,
        model_name=args.gemini_model,
        num_random_contexts=1,
    )

    embedding_provider = EmbeddingProviderFactory.create(
        {
            "type": "process",
            "model": args.embedding_model,
            "workers": args.embedding_workers,
        },
    )

    retriever = QdrantRetriever(
        client=qd,
        collection_name=args.collection,
        embedding_provider=embedding_provider,
        async_client=qd_async,
        payload_key=args.content_key,
        limit=3,
    )
    modifier = WithRAGRespondentPromptModifier(retriever=retriever)

    storage = JSONLStorage(conversations_path=str(args.output))

    conv_gen = ConversationGenerator(
        respondent_prompt=RESPONDENT_PROMPT,
        correspondent_prompt=CORRESPONDENT_PROMPT,
        api_key=api_key,
        model_name=args.gemini_model,
        monitor=monitor,
        auto_improve=args.auto_improve,
        storage=storage,
        instruction_generator_callback=instruction_cb,
        respondent_prompt_modifier=modifier,
        embedding_provider=embedding_provider if args.auto_improve else None,
    )

    print(
        f"Qdrant url={args.qdrant_url!r} collection={args.collection!r} "
        f"content_key={args.content_key!r} embedding_model={args.embedding_model!r}\n"
        f"num_dialogs={args.num_dialogs} max_turns={args.max_turns} "
        f"output={args.output}"
    )

    try:
        await conv_gen.generate(
            num_dialogs=args.num_dialogs,
            max_turns=args.max_turns,
        )
        gen_time = monitor.get_metrics("generation_time", window=timedelta(hours=1))
        if gen_time.get("mean") is not None:
            print(f"Avg. generation time: {gen_time['mean']:.2f}s")
        monitor.visualize_metrics(save_dir=str(args.plots_dir))
    finally:
        await embedding_provider.aclose()
        await qd_async.close()
        monitor.shutdown()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
