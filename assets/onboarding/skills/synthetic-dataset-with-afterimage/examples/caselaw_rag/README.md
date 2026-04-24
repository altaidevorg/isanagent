# Caselaw RAG example (Qdrant + Hugging Face CAP embeddings)

This folder is an end-to-end tutorial: index a small slice of U.S. caselaw text with **precomputed BGE-base-en-v1.5 vectors** from Hugging Face, store it in **Qdrant**, then run **AfterImage** to generate English synthetic dialogs grounded on retrieval.

The source dataset is [free-law/Caselaw_Access_Project_embeddings](https://huggingface.co/datasets/free-law/Caselaw_Access_Project_embeddings) (cleaned opinion text + 768-d embeddings; see the dataset card for licensing and citation).

Outputs are for **research and model training only**, not legal advice.

---

## What you need

- [uv](https://docs.astral.sh/uv/) (recommended) or Python 3.11+
- A **Gemini API key** (`GEMINI_API_KEY`) for generation
- **Qdrant** reachable at a URL you pass to both scripts (local Docker or [Qdrant Cloud](https://cloud.qdrant.io/))
- For **generation** only: install this repo so `afterimage` is importable, with local embeddings for query encoding:

  ```bash
  uv sync --extra embeddings-local
  ```

  Query-time vectors must use the **same model family and dimension** as the index. This example indexes **BAAI/bge-base-en-v1.5** (768-d); `generate.py` defaults to that model.

---

## Step 1 — Run Qdrant

### Option A: Docker (good for laptops)

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

The HTTP API is at `http://localhost:6333`.

### Option B: Qdrant Cloud

1. Create a cluster in the [Qdrant Cloud console](https://cloud.qdrant.io/).
2. Copy the **URL** (e.g. `https://xxxxxx.us-east.aws.cloud.qdrant.io:6333`) and **API key**.
3. Export the key for the scripts:

   ```bash
   export QDRANT_API_KEY="your-api-key"
   ```

---

## Step 2 — Index a streamed slice of the corpus

`index_corpus.py` is a [PEP 723](https://peps.python.org/pep-0723/) script: `uv run` installs `qdrant-client`, `datasets`, and `tqdm` automatically. It **streams** rows so multi-gigabyte Parquet shards are not loaded into RAM.

Quick demo (500 opinions, local Docker):

```bash
uv run examples/caselaw_rag/index_corpus.py \
  --qdrant-url http://localhost:6333 \
  --collection caselaw_cap_demo \
  --num-samples 500 \
  --recreate
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--qdrant-url` | Qdrant HTTP URL (Cloud or Docker) |
| `--qdrant-api-key` | API key (optional if `QDRANT_API_KEY` is set) |
| `--collection` | Collection name (default `caselaw_cap_demo`) |
| `--num-samples` | How many rows to stream from HF (default `500`) |
| `--batch-size` | Upsert batch size (default `128`) |
| `--recreate` | Delete existing collection with the same name, then rebuild |

Indexed payload uses the field **`content`** (text copied from the dataset’s `text` column) so it matches `QdrantRetriever` / `QdrantDocumentProvider` defaults in `generate.py`.

**Larger runs:** raise `--num-samples` as far as you like; Qdrant and disk are the practical limits, not RAM, because of streaming.

**HF authentication:** the dataset is public; if Hugging Face rate-limits you, run `huggingface-cli login` or set `HF_TOKEN`.

---

## Step 3 — Generate conversations

From the **repository root** (with `uv sync --extra embeddings-local` already done):

```bash
export GEMINI_API_KEY="your-key"

uv run python examples/caselaw_rag/generate.py \
  --qdrant-url http://localhost:6333 \
  --collection caselaw_cap_demo \
  --num-dialogs 5 \
  --max-turns 1 \
  --output examples/caselaw_rag/output/conversations.jsonl
```

For **Qdrant Cloud**:

```bash
export GEMINI_API_KEY="your-key"
export QDRANT_API_KEY="your-qdrant-api-key"

uv run python examples/caselaw_rag/generate.py \
  --qdrant-url "https://YOUR-CLUSTER.cloud.qdrant.io:6333" \
  --qdrant-api-key "$QDRANT_API_KEY" \
  --collection caselaw_cap_demo \
  --num-dialogs 10
```

Environment variables mirror the flags when set: `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_CONTENT_KEY`, `QDRANT_MAX_DOCS`, `NUM_DIALOGS`, `MAX_TURNS`, `GEMINI_MODEL`, `EMBEDDING_MODEL`, `CASELAW_OUTPUT`, etc.

**Quality retries:** add `--auto-improve` only if you want the built-in judge loop (adds LLM + embedding cost).

**Artifacts:** JSONL defaults to `examples/caselaw_rag/output/conversations.jsonl`; logs under `examples/caselaw_rag/logs/`; plots under `examples/caselaw_rag/plots/`. These paths are gitignored.

---

## How it fits AfterImage

1. **`index_corpus.py`** — standalone ingestion (HF → Qdrant). Not part of the `afterimage` package.
2. **`generate.py`** — uses `ContextualInstructionGeneratorCallback` + `QdrantDocumentProvider` for instruction-side context, and `QdrantRetriever` + `WithRAGRespondentPromptModifier` for query-time RAG. Retrieval metadata (hit ids/scores) is attached when using the Phase B retriever API.

Session context is fixed **per sampled instruction** (see `docs/conversation_generation.md` for session-scoped vs per-turn RAG).

---

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| `Collection ... already exists` | Pass `--recreate` to `index_corpus.py`, or pick a new `--collection`. |
| Empty retrieval / wrong hits | Confirm `--embedding-model` matches the index (**BAAI/bge-base-en-v1.5**, 768-d). |
| `sentence_transformers` missing | Run `uv sync --extra embeddings-local`. |
| Qdrant connection refused | Docker mapping `6333`, firewall, or Cloud URL including port `6333`. |
| HF download errors | Network, or `HF_TOKEN` for higher rate limits. |

---

## Files

| File | Role |
|------|------|
| `index_corpus.py` | Stream HF dataset → Qdrant (PEP 723 / `uv run`) |
| `generate.py` | AfterImage dialog generation against that collection |
| `README.md` | This tutorial |
