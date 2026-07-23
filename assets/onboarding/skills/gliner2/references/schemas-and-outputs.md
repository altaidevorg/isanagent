# GLiNER2 schemas and outputs

## Contents

- [Choose the schema interface](#choose-the-schema-interface)
- [Quick structured extraction](#quick-structured-extraction)
- [Builder schemas](#builder-schemas)
- [Flat sibling structures](#flat-sibling-structures)
- [Repeated-record cardinality and coherence](#repeated-record-cardinality-and-coherence)
- [Public schema preflight](#public-schema-preflight)
- [Output shapes](#output-shapes)
- [Thresholds and validators](#thresholds-and-validators)
- [Deterministic post-validation](#deterministic-post-validation)

## Choose the schema interface

GLiNER2 exposes two schema styles. Keep their input formats separate.

| Need | Interface |
|---|---|
| Entity-only | `extract_entities(text, entity_types)` |
| Classification-only | `classify_text(text, tasks)` |
| Relation-only | `extract_relations(text, relation_types)` |
| Structure-only, concise definition | `extract_json(text, structures)` |
| Combined tasks, descriptions, per-field settings, validators | `create_schema()` builder then `extract(text, schema)` |
| Validated configuration loaded from JSON/dict | `Schema.from_dict(data)` then `extract` |

Do not pass quick-API field strings to `.structure().field(...)`. Do not pass builder objects where a quick configuration mapping is expected unless the method explicitly accepts a schema.

## Quick structured extraction

`extract_json` accepts a mapping from structure name to a list of field specifications:

```python
result = extractor.extract_json(
    "Phone X costs $799, includes 256GB storage, and is in stock.",
    {
        "product": [
            "name::str::Full product name",
            "price::str::Price with currency",
            "features::list::Product features",
            "availability::[in_stock|pre_order|sold_out]::str::Stock state",
        ]
    },
    include_confidence=True,
    include_spans=True,
)
```

Field-string components are separated by `::`:

- the first component is always the field name;
- `str` selects one value;
- `list` selects multiple values and is the default;
- `[choice_a|choice_b]` defines a closed set and defaults to `str` unless a dtype is supplied;
- any other component is treated as the description.

For programmatically generated schemas, prefer the builder or `Schema.from_dict` over constructing dense field strings.

## Builder schemas

The builder adds task components through chaining:

```python
schema = (
    extractor.create_schema()
    .classification(
        "document_type",
        ["invoice", "credit_note", "purchase_order", "receipt"],
    )
    .entities({
        "company": "Buyer or seller company names",
        "amount": "Monetary amounts with a currency marker",
    })
    .relations({
        "billed_to": "Head is vendor; tail is customer",
    })
    .structure("invoice")
        .field("invoice_number", dtype="str", threshold=0.7)
        .field("vendor", dtype="str")
        .field("customer", dtype="str")
        .field("total", dtype="str")
        .field(
            "payment_status",
            dtype="str",
            choices=["paid", "unpaid", "partial", "overdue"],
        )
    .structure("line_item")
        .field("description", dtype="str")
        .field("quantity", dtype="str")
        .field("unit_price", dtype="str")
        .field("total", dtype="str", threshold=0.3)
)
```

There is no `end_structure()` call. Calling another schema method or `.build()` automatically closes the active structure. `dtype` belongs on `.field(...)` or `.entities(...)`, not on `.structure(...)`.

### Validated dict input

Use the public `Schema.from_dict` for externally supplied schemas:

```python
from gliner2 import Schema

schema = Schema.from_dict({
    "entities": ["company", "person"],
    "structures": {
        "product": {
            "fields": [
                {"name": "name", "dtype": "str"},
                {"name": "features", "dtype": "list"},
                {"name": "availability", "dtype": "str",
                 "choices": ["in_stock", "sold_out"]},
            ]
        }
    },
    "classifications": [
        {"task": "sentiment", "labels": ["positive", "negative", "neutral"]}
    ],
    "relations": ["works_for"],
})
```

`SchemaInput` validates names, non-empty sections, classification label uniqueness, at least two classification labels, dtypes, and choices. The current public dict model is intentionally simpler than every builder option; use the builder when per-field thresholds or regex validators are required.

## Flat sibling structures

Structures are flat record types, not nested object definitions. Model an invoice with repeated rows as two sibling structures:

```text
invoice:   invoice_number, vendor, customer, total
line_item: description, quantity, unit_price, total
```

Do not attempt this unsupported shape:

```text
invoice:
  line_items:
    - description
      quantity
```

The model may return multiple instances of each sibling structure, but it does not emit a guaranteed foreign key between them. This is acceptable when one input contains one invoice: all line items can be associated with that invoice by document scope. When an input contains multiple invoices, split the input first or extract an explicit shared key (for example `invoice_number`) in each sibling record and validate the association. Positional proximity may be a heuristic, never an implicit model guarantee.

Flat sibling records also avoid parallel-list ambiguity. Prefer repeated `line_item` structures over one `order` structure containing separate `items`, `quantities`, and `unit_prices` lists whose indexes may not align.

## Repeated-record cardinality and coherence

A structurally valid repeated result can still undercount events or mix fields
from different events. For example, a document may describe one device
overheating and another device losing connectivity, while the model returns one
`issue` whose component comes from the connectivity clause and whose problem
description comes from the overheating clause. Every span can be correct even
though the assembled record is semantically wrong.

Validate repeated structures at three levels:

1. **Cardinality:** When the input or upstream parser establishes an expected
   number of events, compare it with the returned record count. Missing or extra
   siblings make semantic status at best `PARTIAL`.
2. **Record coherence:** Confirm that fields in one record refer to the same
   subject, event, clause, time, or deterministic grouping key. Do not infer
   coherence merely because every field occurs somewhere in the document.
3. **Coverage:** Confirm that each required event is represented, not only that
   every returned record has all required fields.

Keep the raw model output before normalization or filtering. Report any
deterministic post-processing separately so it cannot hide under-extraction or
cross-event field binding. A lexical rule such as dropping any record containing
`perfectly` is not a general negation solution: it can fail on contrastive text
such as “worked perfectly until yesterday, but now crashes.” Test negation and
contrast scope on representative positive and negative examples.

If a combined schema produces incoherent or missing siblings, first improve the
structure and field descriptions. Then try structure-only extraction, because
separating tasks can reduce competition and simplify error analysis. Tune
field-specific thresholds on held-out data only after checking whether the
missing record received a low score. Consider fine-tuning only when the same
failure remains repeatable across representative labeled data.

## Public schema preflight

Build and inspect a schema before loading model weights. Use public methods only:

```python
from gliner2 import Schema

friendly = schema.to_dict()
assert set(friendly["structures"]) == {"invoice", "line_item"}

invoice_fields = {
    field["name"]: field
    for field in friendly["structures"]["invoice"]["fields"]
}
assert invoice_fields["invoice_number"]["dtype"] == "str"
assert invoice_fields["payment_status"]["choices"] == [
    "paid", "unpaid", "partial", "overdue"
]

# Re-validate the public representation and materialize the inference schema.
Schema.from_dict(friendly)
internal = schema.build()
assert internal["json_structures"]
```

`to_dict()` is the stable, user-friendly representation. `build()` returns the lower-level inference representation. Avoid private attributes such as `_field_metadata`; they are implementation details. The current `to_dict()` round trip does not expose every advanced threshold/validator setting, so test those settings through documented behavior instead of private introspection.

## Output shapes

### Metadata flags

`include_confidence` and `include_spans` are independent:

| Flags | Span-bearing value |
|---|---|
| neither | `"Apple"` |
| confidence | `{"text": "Apple", "confidence": 0.98}` |
| spans | `{"text": "Apple", "start": 0, "end": 5}` |
| both | `{"text": "Apple", "confidence": 0.98, "start": 0, "end": 5}` |

Offsets are Python half-open character indexes: `source[start:end]` must equal `text`.

### Entities

Entity results are under `entities` and grouped by type. `dtype="list"` produces a list; `dtype="str"` produces one value or a missing value. If the model predicts no entity structure at all, the `entities` mapping may be empty, so never equate mapping existence with non-empty extraction.

### Classification

Without confidence, a single-label task returns a string and a multi-label task returns a list of strings. With confidence:

```python
{"sentiment": {"label": "positive", "confidence": 0.91}}
{"topics": [{"label": "technology", "confidence": 0.84}]}
```

Classification values are closed labels and do not have source spans.

### Structures

Each structure name maps to a list of record dictionaries. A `str` field is one value; a `list` field is a list. Missing single fields can be `None`; missing repeated fields can be empty lists. Do not replace either with fabricated defaults before validation.

Choice fields are classification-style values. They can include `text` and `confidence`, but do not carry source offsets even when `include_spans=True`, because the selected canonical label need not occur verbatim in the source.

### Relations

Without metadata, relations are tuples `(head, tail)`. With confidence or spans, each relation is a dictionary containing `head` and `tail`, and those endpoints are recursively span-bearing dictionaries. Requested relation types are grouped under `relation_extraction` and normally retained as empty lists when not found.

## Thresholds and validators

The `threshold` argument to extraction is the default span threshold. Override it narrowly:

```python
schema = (
    extractor.create_schema()
    .entities({
        "email": {"description": "Email addresses", "threshold": 0.9},
        "person": {"description": "Names of people", "threshold": 0.55},
    })
    .relations({
        "authorized_by": {
            "description": "Head action is authorized by tail person",
            "threshold": 0.8,
        }
    })
    .structure("invoice")
        .field("invoice_number", dtype="str", threshold=0.8)
        .field("total", dtype="str", threshold=0.4)
)
```

Do not lower the global threshold to recover one low-confidence field without measuring the new false positives across every other field.

Local inference supports post-extraction regex filters:

```python
from gliner2 import RegexValidator

invoice_id = RegexValidator(r"^INV-\d{4}-\d+$")
schema = (
    extractor.create_schema()
    .structure("invoice")
        .field("invoice_number", dtype="str", validators=[invoice_id])
)
```

All validators on a field must pass. Validators filter spans; they do not repair them. The hosted API builder warns and ignores `RegexValidator`, so repeat critical validation in application code regardless of inference mode.

## Deterministic post-validation

Validate model output before using it as a record of truth:

1. Traverse nested dictionaries and lists; validate every object containing `start`/`end` against the exact input.
2. Check required fields explicitly. A present structure with `None` fields is incomplete.
3. Check every closed choice against the configured set.
4. Parse dates, identifiers, currencies, quantities, and totals with deterministic libraries.
5. Check relation direction and endpoint types.
6. For repeated structures, validate expected record count, event coverage, and within-record field coherence.
7. Decide how to handle duplicate and overlapping spans; offset correctness alone does not resolve semantic conflicts.
8. Preserve raw output and label deterministic post-processing separately.
9. Send low-confidence, contradictory, or high-impact results to review according to policy. Do not tie mandatory review only to empty extraction.

For money, avoid binary floating-point:

```python
from decimal import Decimal
import re

def money(value: str) -> Decimal:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        raise ValueError(f"No amount in {value!r}")
    return Decimal(match.group(0).replace(",", ""))

assert money("USD 75") * Decimal("2") == money("USD 150")
```

The bundled `infer_structured.py` demonstrates public schema preflight, recursive span validation, required-field checks, and invoice arithmetic without using private GLiNER2 state.
