# Preparing GLiNER2 training data

Use this reference when creating, converting, splitting, or validating data for entity extraction, classification, structured extraction, relations, or combined training.

## Contents

- [Canonical JSONL](#canonical-jsonl)
- [Python data classes](#python-data-classes)
- [Task annotations](#task-annotations)
- [Negative and missing examples](#negative-and-missing-examples)
- [Validation](#validation)
- [Relation and classification consistency](#relation-and-classification-consistency)
- [Deterministic splitting and leakage](#deterministic-splitting-and-leakage)
- [Recommended workflow](#recommended-workflow)

## Canonical JSONL

Prefer one UTF-8 JSON object per line with `input` and `output`:

```jsonl
{"input":"Ada works at Northwind in Ankara.","output":{"entities":{"person":["Ada"],"organization":["Northwind"],"location":["Ankara"]}}}
```

The lower-level trainer loader also recognizes `text`/`schema`. Current `TrainingDataset.load()` and `InputExample.from_dict()` directly expect `input`/`output`, however. The bundled scripts normalize either form before constructing `TrainingDataset`; use the canonical form for maximum portability.

Valid annotation keys inside `output` are:

| Key | Shape |
|---|---|
| `entities` | `{entity_type: [mention, ...]}` |
| `entity_descriptions` | `{entity_type: description}` |
| `classifications` | list of classification task objects |
| `json_structures` | list of `{structure_name: {field: value}}` objects |
| `json_descriptions` | `{structure_name: {field: description}}` |
| `relations` | list of `{relation_name: {field: value}}` objects |

Annotations contain mention strings, not character offsets. Strict validation checks non-empty entity, structure, and relation string values with case-insensitive substring matching against the input. Preserve the original text after annotation; normalization that changes mentions can invalidate the data.

## Python data classes

Use public classes from `gliner2.training.data` for programmatic creation:

```python
from gliner2.training.data import (
    ChoiceField,
    Classification,
    InputExample,
    Relation,
    Structure,
    TrainingDataset,
)

example = InputExample(
    text="Order A-17 for two keyboards is approved.",
    entities={"order_id": ["A-17"]},
    classifications=[
        Classification(
            task="approval_status",
            labels=["approved", "rejected", "pending"],
            true_label="approved",
        )
    ],
    structures=[
        Structure(
            "order",
            order_id="A-17",
            quantity="two",
            status=ChoiceField("approved", ["approved", "rejected", "pending"]),
        )
    ],
)

dataset = TrainingDataset([example])
report = dataset.validate(raise_on_error=False)
```

Current validation is always strict. Do not pass the stale tutorial argument `strict=True`; the current signature is `validate(raise_on_error: bool = True)`.

## Task annotations

### Entities

Keys define the schema presented to the model; values are exact mentions:

```json
{
  "entities": {
    "person": ["Ada Lovelace"],
    "organization": ["Analytical Engine Society"]
  },
  "entity_descriptions": {
    "organization": "Named companies, institutions, or associations"
  }
}
```

Use stable, descriptive type names. If descriptions are present, every described type must also be present in `entities` for that example.

### Classification

```json
{
  "classifications": [{
    "task": "ticket_topic",
    "labels": ["billing", "technical", "account"],
    "true_label": ["billing"],
    "multi_label": false,
    "label_descriptions": {"billing": "Invoices, charges, or refunds"}
  }]
}
```

`true_label` may be a string or list for single-label data; GLiNER2 normalizes a string to a one-element list. Always use a list for multi-label data. Every true label and description key must occur in `labels`. Keep label order and `multi_label` stable for the same task across the dataset.

Optional `prompt`, `examples`, and `label_descriptions` fields are supported. Each few-shot item in `examples` must be a two-element input/output pair.

### Structured extraction

Each occurrence is a separate object. Repeat the same parent name for repeated records:

```json
{
  "json_structures": [
    {"line_item": {"description": "keyboard", "quantity": "2", "price": "$75"}},
    {"line_item": {"description": "mouse", "quantity": "3", "price": "$20"}}
  ]
}
```

Field values may be strings, lists of strings, or choices:

```json
{"status": {"value": "approved", "choices": ["approved", "rejected", "pending"]}}
```

The current validator permits different instances of one structure type to contain different field subsets. The training schema uses the union. Keep fields stable when possible, and represent missing scalar span values as `""` only when that absence is intentional.

### Relations

Binary relations use directional `head` and `tail` fields:

```json
{"relations": [{"works_for": {"head": "Ada", "tail": "Northwind"}}]}
```

Custom fields are accepted for training:

```json
{"relations": [{"transfer": {"sender": "Ada", "recipient": "Lin", "amount": "$50"}}]}
```

For each relation name, the first occurrence establishes its field names. All later occurrences of that relation type across the complete dataset must use exactly the same field set. Direction matters: `(head, tail)` and `(tail, head)` are different annotations.

Public inference convenience methods emit binary head/tail relations. Use binary relation data when you need direct task-aware evaluation with `scripts/evaluate_gliner2.py`; it reports custom-field relations as unsupported rather than mis-scoring them.

### Combined examples

One example may contain any combination of the four task types. This is useful when the same text is fully annotated. Do not add fabricated annotations simply to make a record multi-task.

## Negative and missing examples

Each example must retain at least one task schema. A completely empty output is invalid.

An entity-negative example can retain the requested types with empty lists:

```jsonl
{"input":"No person or company is named here.","output":{"entities":{"person":[],"organization":[]}}}
```

This passes the current validator because the entity type keys define a non-empty task. By contrast, `{"entities": {}}` alone has no task and is invalid.

An empty `relations: []` alone does not identify which relation types are negative and is not a valid standalone task. Pair it with a genuinely annotated task only if the text is also used for that task; do not pretend this trains a specific negative relation schema. Similarly, missing structured fields should be explicit only when that representation matches the intended inference schema.

Include hard negatives and empty entity lists deliberately, but measure their proportion. Excessive easy negatives can dominate optimization.

## Validation

Run the bundled validator before model loading:

```bash
uv run /path/to/gliner2/scripts/validate_training_data.py train.jsonl
uv run /path/to/gliner2/scripts/validate_training_data.py \
  train.jsonl validation.jsonl test.jsonl \
  --report artifacts/data-validation.json
```

The script:

- parses every nonblank JSONL line;
- normalizes `text`/`schema` to `input`/`output`;
- calls current `TrainingDataset.validate(raise_on_error=False)`;
- calls `validate_relation_consistency()` across all records;
- checks that one classification task does not change labels/order or cardinality;
- reports exact duplicate records;
- fails on normalized text shared across different input files by default;
- emits JSON and returns non-zero on hard errors.

GLiNER2's trainer-side `validate_data=True` goes through a sanitizing loader that can drop invalid annotations or records. For auditable training, validate first and stop on errors; the bundled training script does this and then disables the sanitizing pass.

## Relation and classification consistency

`InputExample.validate()` catches inconsistent relation fields only within one example. `TrainingDataset.validate_relation_consistency()` is required to check the full dataset. Run it on the union of train, validation, and test, not on each split independently.

The current library does not enforce cross-example classification schema identity. The bundled validator adds that guardrail because changing candidate labels or `multi_label` under the same task name makes evaluation and training ambiguous.

Also inspect dataset statistics:

```python
stats = dataset.stats()
dataset.print_stats()
```

Review task distribution, label balance, type frequency, text lengths, empty tasks, and unusually repeated mentions. Validation establishes structural correctness, not annotation correctness.

## Deterministic splitting and leakage

For a one-time random split:

```python
train, validation, test = dataset.split(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle=True,
    seed=42,
)
train.save("train.jsonl")
validation.save("validation.jsonl")
test.save("test.jsonl")
```

Ratios must sum to one. Integer truncation can make tiny validation sets empty, so assert split sizes explicitly.

Random row splitting is unsafe when documents produce multiple rows or near-duplicates. Prefer grouping by source document, customer, patient, thread, or another leakage boundary, then assign whole groups deterministically. Keep the test split untouched until the final evaluation. Record the split seed, grouping key, input hashes, and dataset revision.

The bundled validator detects exact normalized-text overlap across files. It cannot detect paraphrases, shared document fragments, or label leakage; audit these separately.

## Recommended workflow

1. Define stable task schemas and annotation guidelines.
2. Create canonical JSONL and retain source/document IDs outside the model text when possible.
3. Validate the complete corpus and fix every hard error.
4. Split by leakage-safe groups with a fixed seed.
5. Validate all splits together to catch cross-split schema differences and text overlap.
6. Freeze the test set.
7. Establish a base-model evaluation before fine-tuning.
8. Run one optimizer-step smoke training.
9. Run a bounded experiment and inspect both loss and task-aware validation metrics.
10. Only then schedule full training.
