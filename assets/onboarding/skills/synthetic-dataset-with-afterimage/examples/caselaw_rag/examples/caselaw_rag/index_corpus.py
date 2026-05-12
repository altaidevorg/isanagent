# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "qdrant-client>=1.7.0",
#     "datasets>=2.16.0",
#     "tqdm>=4.66.0",
# ]
# ///
"""Stream Caselaw Access Project embeddings from Hugging Face into Qdrant.

Uses precomputed **BGE-base-en-v1.5** vectors (768-dim) from
``free-law/Caselaw_Access_Project_embeddings`` so no local embedding run is
required for indexing.

Run (dependencies are installed automatically by ``uv``)::

    uv run examples/caselaw_rag/index_corpus.py --num-samples 500 --qdrant-url http://localhost:6333

See ``README.md`` in this directory for the full tutorial.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import chain, islice

from typing import Any, Iterator
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

DEFAULT_DATASET = "free-law/Caselaw_Access_Project_embeddings"
EXPECTED_DIM = 768
TEXT_FIELD = "text"
VECTOR_FIELD = "embeddings"
PAYLOAD_TEXT_KEY = "content"


def _as_vector(raw: Any) -> list[float]:
    if raw is None:
        raise ValueError("missing embedding")
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [float(x) for x in raw]


def _stream_limited(dataset_id: str, split: str, limit: int) -> Iterator[dict[str, Any]]:
    
    ds = load_dataset(dataset_id, split=split, streaming=True)
    yield from islice(ds, limit)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant HTTP URL",
    )
    p.add_argument(
        "--qdrant-api-key",
        default=None,
        help="Optional API key (Qdrant Cloud). Overrides QDRANT_API_KEY.",
    )
    p.add_argument(
        "--collection",
        default="caselaw_cap_demo",
        help="Collection name to create or recreate",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of dataset rows to stream and index",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Points per upsert batch",
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset id",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split name",
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the collection if it exists before creating",
    )
    args = p.parse_args()

    api_key = args.qdrant_api_key or os.environ.get("QDRANT_API_KEY")

    if args.num_samples < 1:
        print("--num-samples must be >= 1", file=sys.stderr)
        return 2

    client = QdrantClient(url=args.qdrant_url, api_key=api_key, timeout=120.0)

    base_it = _stream_limited(args.dataset, args.split, args.num_samples)
    try:
        first = next(base_it)
    except StopIteration:
        print("Dataset stream returned no rows.", file=sys.stderr)
        return 3

    if TEXT_FIELD not in first or VECTOR_FIELD not in first:
        print(
            f"Expected columns {TEXT_FIELD!r} and {VECTOR_FIELD!r}; "
            f"got keys: {list(first.keys())}",
            file=sys.stderr,
        )
        return 4
    try:
        vec0 = _as_vector(first[VECTOR_FIELD])
    except Exception as e:
        print(f"Invalid first embedding: {e}", file=sys.stderr)
        return 4
    if len(vec0) != EXPECTED_DIM:
        print(
            f"Expected embedding dim {EXPECTED_DIM}, got {len(vec0)}.",
            file=sys.stderr,
        )
        return 5

    if client.collection_exists(args.collection):
        if not args.recreate:
            print(
                f"Collection {args.collection!r} already exists. "
                "Pass --recreate to replace it, or choose another --collection.",
                file=sys.stderr,
            )
            return 6
        client.delete_collection(args.collection)

    client.create_collection(
        collection_name=args.collection,
        vectors_config=qm.VectorParams(
            size=EXPECTED_DIM,
            distance=qm.Distance.COSINE,
        ),
    )

    batch: list[qm.PointStruct] = []
    point_id = 0

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        client.upsert(collection_name=args.collection, points=batch, wait=True)
        batch = []

    full_iter = chain([first], base_it)
    for row in tqdm(
        full_iter,
        total=args.num_samples,
        desc="Indexing",
        unit="row",
    ):
        try:
            text = row.get(TEXT_FIELD)
            if not text or not str(text).strip():
                continue
            vec = _as_vector(row.get(VECTOR_FIELD))
            if len(vec) != EXPECTED_DIM:
                continue
        except Exception:
            continue
        batch.append(
            qm.PointStruct(
                id=point_id,
                vector=vec,
                payload={PAYLOAD_TEXT_KEY: str(text)[:120_000]},
            )
        )
        point_id += 1
        if len(batch) >= args.batch_size:
            flush()

    flush()
    count = client.count(args.collection, exact=True).count
    print(
        f"Done: collection={args.collection!r} vector_count={count} "
        f"(qdrant_url={args.qdrant_url!r})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
