# GLiNER2 Usage Reference

Use this reference when building an application with GLiNER2. It describes the public patterns needed for extraction; it does not require access to the GLiNER2 source repository.

## Select local or API inference

Install only schema validation and API support:

```bash
pip install gliner2
```

Install local model inference and training dependencies:

```bash
pip install "gliner2[local]"
```

Load a local model:

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
```

Use the hosted API by setting `PIONEER_API_KEY` and constructing the same high-level interface:

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_api()
```

Pass `api_key=` directly only when environment-based secret injection is unavailable. Never hard-code or log API keys. Use local inference for offline/privacy-sensitive work; use API inference when avoiding model downloads or local ML resources matters.

Local and hosted extractors share the common quick APIs, but they are not capability-identical. The hosted client described by this reference does not expose `extract_long`, `batch_extract_long`, `extract_entities_long`, or `batch_extract_entities_long`. Before using a less common method, verify it on the actual object:

```python
import inspect

method = getattr(extractor, "extract_entities_long", None)
if method is None:
    raise RuntimeError("This extractor does not support long-document extraction")
print(inspect.signature(method))
```

Do not offer a hosted/local runtime switch around code that later calls a local-only method. For sensitive long documents, prefer local inference. If an approved hosted deployment must handle long input, use a separately designed and tested application-side chunking pipeline rather than silently truncating.

### Install repository source for native long-context methods

Do not infer long-context support from `gliner2.__version__` alone. The PyPI `1.3.2` wheel was built before the repository's long-context implementation while later repository source still reports `1.3.2`. First inspect the exact import:

```bash
python - <<'PY'
import gliner2
from gliner2 import GLiNER2

methods = (
    "extract_long",
    "batch_extract_long",
    "extract_entities_long",
    "batch_extract_entities_long",
)
print("version:", gliner2.__version__)
print("path:", gliner2.__file__)
for name in methods:
    print(name, hasattr(GLiNER2, name))
PY
```

If a required method is missing and source installation is authorized, create an isolated environment and install an immutable repository revision. Commit `31c8abaa4a6d88ae8bb6f2e63cfea9926956497c` is the source baseline verified by this reference:

```bash
python -m venv .venv-gliner2-source
source .venv-gliner2-source/bin/activate
python -m pip install --upgrade pip
python -m pip install "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"
```

When a trusted checkout of that revision is already available, an editable install is also valid:

```bash
python -m pip install -e "/path/to/GLiNER2[local]"
```

Run the capability probe again in the same interpreter. Record the virtual-environment Python path, `gliner2.__file__`, version, install command, and commit. Loading the public model is a separate step and normally needs no GLiNER2 API key; it needs model-download access unless the weights are cached. A GPU is optional for a bounded smoke test and beneficial for larger workloads.

Do not attach missing methods dynamically, copy a fallback implementation into the test, or label application-side chunking as native GLiNER2 behavior. Test such fallbacks only when explicitly requested and report them in a separate result category.

## Design schemas deliberately

- Use concise domain labels such as `company`, `medication`, or `effective_date`.
- Add descriptions when labels are ambiguous or domain-specific.
- Use `dtype="str"` for one best value and `dtype="list"` for multiple values.
- Use choices for closed categories rather than extracting arbitrary text.
- Start with the fewest labels needed, inspect errors, then refine descriptions and thresholds.
- Treat model output as probabilistic. Validate critical values with application logic.

## Extract entities

Use the quick API for a single entity task:

```python
result = extractor.extract_entities(
    "Apple CEO Tim Cook announced Vision Pro in Cupertino.",
    {
        "company": "Company or organization names",
        "person": "Names of people",
        "product": "Commercial product names",
        "location": "Cities or physical locations",
    },
    threshold=0.5,
    include_confidence=True,
    include_spans=True,
)
```

The result is grouped under `entities`. With both flags enabled, each item resembles:

```python
{"text": "Tim Cook", "confidence": 0.92, "start": 10, "end": 18}
```

Spans use half-open character offsets. Verify them when correctness matters:

```python
for items in result["entities"].values():
    for item in items:
        assert text[item["start"]:item["end"]] == item["text"]
```

This assertion proves offset integrity only. It does not prove that the model selected the correct entity or boundary; compare results with annotated examples for semantic correctness.

For schema composition, use:

```python
schema = extractor.create_schema().entities(
    {"drug": "Medication names", "dosage": "Amounts such as 500mg"},
    dtype="list",
)
result = extractor.extract(text, schema, include_confidence=True)
```

## Classify text

Use a single-label classification when exactly one category should win:

```python
result = extractor.classify_text(
    "The service was fast and helpful.",
    {"sentiment": ["positive", "negative", "neutral"]},
)
```

Use multi-label mode when several labels can apply:

```python
result = extractor.classify_text(
    "The smartwatch adds health features and raised the company's stock.",
    {
        "topics": {
            "labels": ["technology", "business", "health", "sports"],
            "multi_label": True,
            "cls_threshold": 0.3,
        }
    },
    include_confidence=True,
)
```

Descriptions can replace bare label strings when category meaning is subtle. Tune `cls_threshold` against representative validation examples rather than assuming one universal value.

The schema builder uses a different multi-label form. Pass configuration as keyword arguments:

```python
schema = (
    extractor.create_schema()
    .classification("sentiment", ["positive", "negative", "neutral"])
    .classification(
        "topics",
        ["technology", "business", "health", "sports"],
        multi_label=True,
        cls_threshold=0.3,
    )
)
```

Do not write this builder call:

```python
# Wrong: this creates labels named "labels" and "multi_label".
schema.classification(
    "topics",
    {"labels": ["technology", "health"], "multi_label": True},
)
```

Dictionary labels passed directly to the builder mean label descriptions, for example `{"invoice": "A bill requesting payment"}`. The `{labels, multi_label, cls_threshold}` wrapper belongs to `classify_text` and `batch_classify_text`.

## Extract structured JSON

Use `extract_json` for structure-only work. A field defaults to a list; append `::str` for one value, `::list` for multiple values, and `[a|b|c]` for choices.

```python
result = extractor.extract_json(
    "MacBook Pro costs $1999 with an M3 chip and 16GB RAM.",
    {
        "product": [
            "name::str::Product name",
            "price::str::Advertised price",
            "features::list::Hardware or software features",
        ]
    },
)
```

Structures return lists because multiple instances may occur. Do not assume the first structure is the only one.

For richer composition, use the builder:

```python
schema = (
    extractor.create_schema()
    .structure("reservation")
        .field("restaurant", dtype="str")
        .field("date", dtype="str")
        .field("party_size", dtype="str")
        .field("seating", dtype="str", choices=["indoor", "outdoor", "bar"])
)
result = extractor.extract(text, schema)
```

GLiNER2 structures are flat named records, not arbitrary nested object graphs. Represent repeated line items as repeated instances of a sibling `line_item` structure. If downstream code must associate line items with one of several invoices in the same input, split the input or add deterministic association logic; do not assume nested parent-child grouping that the schema does not express.

Use sibling structures for an invoice and its repeated line items:

```python
schema = (
    extractor.create_schema()
    .structure("invoice")
        .field("invoice_number", dtype="str")
        .field("vendor_name", dtype="str")
        .field("customer_name", dtype="str")
        .field("issue_date", dtype="str")
        .field("due_date", dtype="str")
        .field("currency", dtype="str")
        .field(
            "payment_status",
            dtype="str",
            choices=["paid", "unpaid", "partial", "overdue"],
        )
    .structure("line_item")
        .field("description", dtype="str")
        .field("quantity", dtype="str")
        .field("unit_price", dtype="str")
        .field("total", dtype="str")
)
```

The builder auto-finishes the active structure when another schema method is chained or `build()` is called. There is no `end_structure()` method. `structure()` does not accept `dtype`; cardinality applies to fields, while a named structure already returns a list of instances.

Reject these invented forms during review:

```python
# Wrong: structure() has no dtype argument.
schema.structure("line_item", dtype="list")

# Wrong: structures cannot be nested through the builder.
schema.structure("invoice").structure("line_item")

# Wrong: this method does not exist.
schema.end_structure()
```

Extract numbers and money as text, then validate deterministically. Use `decimal.Decimal`, explicit currency parsing, locale rules, and rounding tolerances for accounting; avoid binary `float` equality.

## Extract relations

```python
result = extractor.extract_relations(
    "John works for Apple and lives in San Francisco.",
    {
        "works_for": "Employment from a person to an organization",
        "lives_in": "Residence from a person to a location",
    },
    include_confidence=True,
    include_spans=True,
)
```

Results are grouped under `relation_extraction`. Without metadata, values are `(source, target)` pairs. Requested relation labels remain present with empty lists when nothing matches. Use directional descriptions because `source` and `target` order matters.

With confidence and spans enabled, relation items use `head` and `tail` endpoint objects:

```python
{
    "relation_extraction": {
        "works_for": [{
            "head": {"text": "John", "confidence": 0.91, "start": 0, "end": 4},
            "tail": {"text": "Apple", "confidence": 0.89, "start": 15, "end": 20},
        }]
    }
}
```

Endpoints do not contain entity-type labels such as `source.label`. Test direction with annotated expected head/tail text and spans. If type constraints are required, run or combine entity extraction and validate the endpoints against those independently extracted entity types.

## Combine tasks in one schema

Compose entities, classifications, relations, and structures when they analyze the same text:

```python
schema = (
    extractor.create_schema()
    .entities({
        "person": "People mentioned",
        "company": "Companies mentioned",
        "product": "Products mentioned",
    })
    .classification("document_type", ["news", "review", "support_request"])
    .classification(
        "topics",
        ["technology", "business", "support"],
        multi_label=True,
        cls_threshold=0.3,
    )
    .relations({"works_for": "Person employed by company"})
)

result = extractor.extract(text, schema, include_confidence=True)
```

Prefer one combined call when tasks share the same input and latency matters. Keep separate calls when tasks need different preprocessing, thresholds, audit rules, or failure handling.

Formatted combined results use task names at the top level. Expect keys such as `entities`, `document_type`, `topics`, `relation_extraction`, and each structure name such as `issue_record`; do not assume generic `classification` or `structure` wrapper keys. Validate the actual output contract before integrating it.

## Process long documents

Standard extraction with `max_len` truncates. Use explicit long-context methods to scan the full document:

```python
result = extractor.extract_entities_long(
    long_text,
    {"company": "Company names", "effective_date": "Contract effective dates"},
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
    include_confidence=True,
)
```

Use `extract_long(long_text, schema, ...)` for a composed schema, `batch_extract_entities_long` for several documents with one entity schema, and `batch_extract_long` for generic schemas.

When the requirement is multiple long documents, show and use the batch API rather than only discussing it:

```python
results = extractor.batch_extract_long(
    contract_texts,
    schema,
    batch_size=8,
    chunk_size=512,
    chunk_overlap=128,
    threshold=0.5,
    include_confidence=True,
    include_spans=True,
)

assert len(results) == len(contract_texts)
```

Confirm that the native batch method is actually invoked and that changing `batch_size` reaches its batching path. A Python loop over single-document calls—even when named `batch_extract_long`—is sequential fallback processing, not evidence of batch inference.

- Keep `chunk_overlap < chunk_size`.
- Increase overlap when relevant phrases may cross chunk boundaries.
- Expect greater overlap to increase work.
- Returned spans are global offsets into the original document.
- Overlap duplicates are merged, but identical text at distinct document positions remains distinct.
- Set `include_spans=True` whenever downstream code reads `start` or `end`; `include_confidence=True` alone does not include offsets.
- Pass the intended `threshold` in the call. Describing a threshold in comments does not configure it.
- Preserve the exact processed text. If OCR normalization changes text, retain that inference text and an offset map back to the source artifact when source-coordinate auditing is required.
- These long-document methods apply to the local extractor described here. Verify hosted-client support instead of assuming parity.

## Batch and validate outputs

Use batch methods when processing many inputs with compatible schemas. Preserve the association between each input and its output, especially when schemas vary. During development:

1. Keep a small representative evaluation set with expected entities, relations, structures, and classifications.
2. Request confidence and spans for inspection.
3. Assert every span maps back to the exact source substring.
4. Check empty, missing, repeated, and ambiguous cases.
5. Measure precision and recall while tuning descriptions and thresholds.
6. Pin the `gliner2` package and model version in production.

For test reports, assign exactly one overall status to each scenario. `PASS` means every explicit requirement and assertion succeeded; use `PARTIAL` when the code executes but a required prediction, field, test branch, or behavior is missing. Do not add subtest statuses to scenario totals. Report two dimensions when useful: API/implementation validity and semantic extraction quality. Correct output shape and spans do not establish that the model extracted the right facts.

Pin versions from evidence, not examples. Record `gliner2.__version__`, the exact model identifier or local revision, schema revision, and preprocessing revision. Never invent a version such as a future-looking placeholder and present it as installed.

Do not assume `GLiNER2.from_pretrained` has Hugging Face `from_pretrained` semantics for arbitrary keyword arguments. In the repository implementation covered by this reference, a supplied `revision=` remains unused and does not reach model/config/tokenizer downloads. For immutable model pinning:

1. Resolve and download an approved model snapshot at a specific commit using deployment tooling.
2. Verify the snapshot contents or record its immutable revision/hash.
3. Pass the verified local snapshot directory to `GLiNER2.from_pretrained(local_snapshot_path)`.
4. Record the installed `gliner2.__version__`, snapshot revision/hash, schema revision, and preprocessing revision.

If a newer installed implementation may forward `revision`, inspect and test that behavior before relying on it. Never use `"main"` as evidence of immutable pinning.

For static validation that does not require model weights, build and inspect the schema:

```python
built = schema.build()
topics = next(item for item in built["classifications"] if item["task"] == "topics")
assert topics["labels"] == ["technology", "business", "health", "sports"]
assert topics["multi_label"] is True
assert topics["cls_threshold"] == 0.3
```

Also validate structures before presenting builder code:

```python
built = schema.build()
structure_names = [next(iter(item)) for item in built["json_structures"]]
assert structure_names == ["invoice", "line_item"]

invoice = built["json_structures"][0]["invoice"]
assert set(invoice) >= {
    "invoice_number",
    "vendor_name",
    "customer_name",
    "issue_date",
    "due_date",
    "currency",
    "payment_status",
}
```

If schema construction or `build()` raises, fix the code instead of labeling it representative. This preflight needs neither model weights nor a GPU.

Validate all returned spans recursively, not only entity lists:

```python
def iter_span_items(value):
    if isinstance(value, dict):
        if {"text", "start", "end"} <= value.keys():
            yield value
        for child in value.values():
            yield from iter_span_items(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_span_items(child)


def assert_spans_match(text, result):
    for item in iter_span_items(result):
        start, end = item["start"], item["end"]
        assert 0 <= start <= end <= len(text)
        assert text[start:end] == item["text"]
```

This covers entity items, relation `head`/`tail` endpoints, and span-bearing structured fields.

Requested entity labels can produce a non-empty mapping whose values are all empty lists. Detect a truly empty extraction with:

```python
entities = result.get("entities", {})
has_any_entity = any(items for items in entities.values())
requires_review = not has_any_entity
```

Do not use only `not entities` for this decision.

This example expresses a completeness fallback, not a universal human-review policy. When the application requires every medical, legal, financial, or other high-impact result to be reviewed, set `requires_review = True` independently of whether entities were found. Track separate reasons such as `missing_required_fields`, `contradictory_values`, `invalid_dosage`, or `low_confidence`; never treat a non-empty extraction as verified fact.

Only call code verified after it has actually run. Otherwise label it representative and list the untested dependencies and calls.

## Handle failures safely

Use the public exception classes:

```python
from gliner2 import (
    AuthenticationError,
    GLiNER2APIError,
    ServerError,
    ValidationError,
)

try:
    result = extractor.extract_entities(text, labels)
except AuthenticationError:
    # Invalid or expired credentials: do not retry.
    raise
except ValidationError:
    # Invalid request/schema: fix the request rather than retrying it.
    raise
except ServerError:
    # Apply service policy after accounting for client retries.
    raise
except GLiNER2APIError:
    # Timeout, connection, malformed response, or another API failure.
    raise
```

The hosted client already configures retries with backoff for POST requests returning `429`, `500`, `502`, `503`, or `504`. Do not add an outer retry loop without calculating the resulting total attempts and latency. Prefer exception types and `status_code` over parsing strings such as `"401"` from messages.

- Catch GLiNER2 API errors at the application boundary.
- Configure explicit timeouts appropriate to the service.
- Do not retry permanent authentication or schema errors.
- Validate extracted dates, identifiers, monetary values, and regulated-domain content before downstream actions.
- Do not treat confidence as a calibrated probability without measuring it on the target domain.
- Avoid sending sensitive text to a hosted API unless policy permits it; choose local inference when necessary.
- Do not log source text, secrets, extracted sensitive values, or review-reason strings containing them. Log opaque document identifiers, counts, timings, error categories, and policy-approved metadata.
- Local inference avoids transmitting text to the hosted service but does not by itself establish HIPAA, GDPR, or other regulatory compliance.

Treat stdout and `print()` as logs. Fields such as `first_company`, `first_amount_text`, medication names, dosage strings, clause text, and extracted spans are sensitive values rather than aggregate metadata. Keep them out of generic logs even when only the “first” value is emitted.

For tests, prefer dependency injection over a module-level API client:

```python
def extract_ticket(extractor, text, labels):
    return extractor.extract_entities(text, labels)
```

Create or patch the extractor actually passed to the function. Patching `GLiNER2.from_api` after a global extractor was already constructed does not replace that global object and can allow an unintended real request.

Mock at the correct layer for the property under test. Replacing `requests.Session.post` is useful for preventing an external call, but it bypasses the mounted HTTP adapter and therefore does not exercise adapter retry/backoff behavior. Test retry configuration by inspecting the mounted adapter or with a controlled transport/response fixture that the adapter processes. Keep missing-key, authentication, validation, transient, permanent, and no-secret-logging checks as explicit separate assertions.

## Check version-specific behavior

When code fails against an installed version, inspect rather than guess:

```python
import inspect
import gliner2
from gliner2 import GLiNER2

print(gliner2.__version__)
print(inspect.signature(GLiNER2.from_pretrained))
print(inspect.signature(GLiNER2.extract_entities))
```

Prefer documented public imports and methods. Do not depend on `gliner2.model`, `gliner2.processor`, or other internals in consumer applications unless accepting version-coupling intentionally.

Use syntax compatible with the application's declared Python runtime. GLiNER2 may support older Python versions than annotations such as `list[str]`; use `typing.List` when targeting Python 3.8.
