# GLiNER2 inference

## Contents

- [Install and load](#install-and-load)
- [Named entities](#named-entities)
- [Classification](#classification)
- [Relations](#relations)
- [Combined schemas](#combined-schemas)
- [Batch inference](#batch-inference)
- [Long documents](#long-documents)
- [Hosted API](#hosted-api)
- [Validation workflow](#validation-workflow)

Read `schemas-and-outputs.md` when designing structured records or consuming metadata-rich results. Read `concepts-and-models.md` before choosing a model, language strategy, or execution device.

## Install and load

Use an isolated uv-managed environment. Add local inference support to an existing project:

```bash
uv add "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"
```

Or run the bundled PEP 723 scripts without editing a project:

```bash
uv run scripts/infer_entities.py --help
uv run scripts/infer_entities.py --smoke-test --device cpu
uv run scripts/infer_structured.py --smoke-test --device cpu
```

Load a public local model without an API key:

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained(
    "fastino/gliner2-base-v1",
    map_location="cpu",
)
extractor.eval()
print(sorted({str(p.device) for p in extractor.parameters()}))
```

The first load may download model artifacts. Subsequent offline use requires those artifacts in the cache or a verified local model directory. For CUDA or Apple Silicon, pass `map_location="cuda"` or `"mps"` and verify the actual parameter device.

## Named entities

Use the quick method for entity-only work. A list supplies label names; a mapping adds descriptions that disambiguate domain labels.

```python
text = "Apple CEO Tim Cook introduced Vision Pro in Cupertino."
result = extractor.extract_entities(
    text,
    {
        "company": "Companies and organizations",
        "person": "Names of people",
        "product": "Commercial products",
        "location": "Cities and physical locations",
    },
    threshold=0.5,
    include_confidence=True,
    include_spans=True,
)
```

Expected shape:

```python
{
    "entities": {
        "company": [
            {"text": "Apple", "confidence": 0.99, "start": 0, "end": 5}
        ],
        # Every requested type is normally represented; a type may be [].
    }
}
```

Use the builder when one type needs a different cardinality or threshold:

```python
schema = extractor.create_schema().entities({
    "invoice_number": {
        "description": "The document's invoice identifier",
        "dtype": "str",
        "threshold": 0.75,
    },
    "amount": {
        "description": "Monetary amounts with currency",
        "dtype": "list",
        "threshold": 0.55,
    },
})
result = extractor.extract(text, schema, include_spans=True)
```

`dtype="list"` returns all accepted mentions; `dtype="str"` returns the best accepted mention or an empty/`None` value depending on metadata flags. Tune thresholds on labeled data. Do not lower the global threshold solely to recover one weak field when a per-type threshold expresses the intended policy.

## Classification

Single-label tasks use mutually exclusive labels:

```python
result = extractor.classify_text(
    "The service was slow and the order arrived damaged.",
    {"sentiment": ["positive", "negative", "neutral"]},
    include_confidence=True,
)
# {"sentiment": {"label": "negative", "confidence": ...}}
```

For non-exclusive labels, pass a configuration dictionary with `labels`, `multi_label`, and `cls_threshold`:

```python
result = extractor.classify_text(
    "The camera is excellent, but battery life is poor.",
    {
        "aspects": {
            "labels": ["camera", "battery", "display", "performance"],
            "multi_label": True,
            "cls_threshold": 0.4,
        }
    },
    include_confidence=True,
)
# {"aspects": [{"label": "camera", "confidence": ...}, ...]}
```

Use the builder to add descriptions or combine tasks:

```python
schema = extractor.create_schema().classification(
    "document_type",
    {
        "invoice": "A bill requesting payment",
        "receipt": "Evidence of completed payment",
        "contract": "An agreement containing obligations",
    },
)
```

`threshold` on `extract` is the default span threshold. Classification selection uses each classification's `cls_threshold`. Choice fields inside structures use their field threshold.

## Relations

Relations are directional. Define the direction in the label or description and test it explicitly.

```python
schema = extractor.create_schema().relations({
    "works_for": "Head is a person; tail is their employing organization",
    "located_in": "Head is an organization; tail is its location",
})
result = extractor.extract(
    "Tim Cook works for Apple, which is located in Cupertino.",
    schema,
    include_confidence=True,
    include_spans=True,
)
```

With metadata enabled, the output is grouped under `relation_extraction`:

```python
{
    "relation_extraction": {
        "works_for": [{
            "head": {"text": "Tim Cook", "confidence": ..., "start": 0, "end": 8},
            "tail": {"text": "Apple", "confidence": ..., "start": 19, "end": 24},
        }],
        "located_in": [...],
    }
}
```

Without confidence/spans, relation instances are `(head_text, tail_text)` tuples. Requested relation names are retained with empty lists when no relation is accepted. A relation score does not prove direction or factuality; validate both on domain data.

## Combined schemas

Compose entities, classifications, relations, and sibling structures in one schema:

```python
schema = (
    extractor.create_schema()
    .entities({
        "person": "Names of people",
        "company": "Companies and organizations",
    })
    .classification("sentiment", ["positive", "negative", "neutral"])
    .relations({"works_for": "Person to employing organization"})
    .structure("product")
        .field("name", dtype="str")
        .field("price", dtype="str")
        .field("features", dtype="list")
)

# Public, model-free schema inspection before inference:
print(schema.to_dict())
schema.build()

result = extractor.extract(
    "Tim Cook works for Apple. Apple launched Phone X for $799.",
    schema,
    include_confidence=True,
    include_spans=True,
)
```

See `schemas-and-outputs.md` for flat structure semantics and public preflight assertions.

## Batch inference

Use native batch methods for multiple inputs. Do not replace them with a loop that accepts but ignores `batch_size`.

```python
texts = [
    "Apple CEO Tim Cook announced Vision Pro in Cupertino.",
    "Microsoft CEO Satya Nadella announced Copilot in Seattle.",
]
results = extractor.batch_extract_entities(
    texts,
    ["company", "person", "product", "location"],
    batch_size=8,
    include_confidence=True,
    include_spans=True,
)
assert len(results) == len(texts)
```

Available convenience methods include `batch_extract_entities`, `batch_classify_text`, `batch_extract_json`, and `batch_extract_relations`. Use `batch_extract(texts, schema, ...)` for a shared combined schema. Local `batch_extract` also accepts one schema per text when the schema list length equals the text list length.

Preserve input/output order with `zip(texts, results)`. Compare native batch results with single-item results on a small fixture before increasing batch size. `num_workers` is exposed on the generic local `batch_extract`, while convenience batch methods use the default preprocessing worker count.

## Long documents

`max_len=N` truncates ordinary extraction to the first N word tokens. It does not scan the entire document. Use explicit long-context methods when supported by the installed local package:

```python
required = (
    "extract_long",
    "batch_extract_long",
    "extract_entities_long",
    "batch_extract_entities_long",
)
missing = [name for name in required if not hasattr(extractor, name)]
if missing:
    raise RuntimeError(f"Installed GLiNER2 lacks native long-context methods: {missing}")

result = extractor.extract_entities_long(
    long_text,
    {"party": "Contract parties", "effective_date": "Contract start dates"},
    chunk_size=384,
    chunk_overlap=64,
    batch_size=8,
    include_confidence=True,
    include_spans=True,
)
```

The native path splits into overlapping word chunks, batches chunk inference, remaps chunk-local offsets to global document character offsets, and merges duplicate overlap detections. Require `0 <= start <= end <= len(long_text)` and `long_text[start:end] == text` for every returned span.

Constraints:

- Keep `0 <= chunk_overlap < chunk_size`.
- Increase overlap for mentions or relations likely to cross boundaries; measure latency and duplicate behavior.
- Use `batch_extract_entities_long` for multiple documents with one entity schema.
- Use `batch_extract_long` for combined schemas or a schema list.
- Current long methods require `format_results=True`.
- Choice/classification aggregation across chunks is heuristic; evaluate document-level behavior.
- Do not monkeypatch missing methods and call the result native support. Resolve the installed build as described in `troubleshooting.md`.

## Hosted API

Hosted inference requires credentials and sends text to the configured service:

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_api()  # reads PIONEER_API_KEY
result = extractor.extract_entities(text, ["person", "company"])
```

Handle the public exception hierarchy:

```python
from gliner2 import AuthenticationError, GLiNER2APIError, ServerError, ValidationError

try:
    result = extractor.extract_entities(text, ["person"])
except AuthenticationError:
    ...  # invalid or expired key
except ValidationError:
    ...  # request rejected as invalid
except ServerError:
    ...  # provider-side 5xx response
except GLiNER2APIError:
    ...  # timeout, connection, response decoding, or other API failure
```

The client retries POST requests for 429 and selected 5xx responses through its HTTP adapter. The current API client does not expose native long-context methods, ignores `RegexValidator` with a warning, and processes different schemas per batch item sequentially. Do not log the API key or authorization headers.

## Validation workflow

For each inference pipeline:

1. Build and inspect the schema without loading a model where possible.
2. Run a small positive, negative, ambiguous, and empty example set.
3. Request confidence and spans during development.
4. Recursively validate every returned span against the exact original text.
5. Validate choice membership, relation direction, record counts, required fields, types, and identifiers deterministically.
6. Compare single and native batch results and verify ordering.
7. Tune thresholds on held-out data; do not tune and report quality on the same example.
8. Report API/shape validity separately from semantic quality.

Use `scripts/infer_entities.py` and `scripts/infer_structured.py` as executable starting points. They emit JSON, verify actual device and package path, and perform recursive span checks.
