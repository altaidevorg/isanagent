---
name: gliner2
description: Use GLiNER2 for local or hosted named-entity and relation extraction, schema-driven structured output, text classification, combined inference, dataset preparation, evaluation, full fine-tuning, and LoRA/PEFT adapters. Use when an agent needs to select a GLiNER2 model, design or validate extraction schemas, run inference, interpret outputs, prepare GLiNER2 JSONL data, specialize a model for a domain, train or load adapters, or diagnose GLiNER2 consumer-workflow failures.
---

# Use GLiNER2

GLiNER2 is an encoder-based, schema-conditioned information-extraction system. One unified model can recognize named entities, classify text, extract directional relations, and return schema-driven structured records. It can compose these tasks in one extraction call without using a separate model for each task.

Use GLiNER2 when the output is defined by labels, choices, relations, or fields grounded in an input document. Do not treat it as a general-purpose generative model: validate extracted spans, required fields, closed choices, identifiers, dates, and numeric values in application code.

This skill is self-contained for consuming and specializing GLiNER2. It does not
require a GLiNER2 repository checkout and must not route ordinary work into
repository maintenance. The pinned Git dependency below is an installation
source, not a requirement to inspect or modify repository code.

## Select a documented model

| Model | Parameters | Access | Default use |
|---|---:|---|---|
| `fastino/gliner2-base-v1` | 205M | Local | Default inference, evaluation, and fine-tuning |
| `fastino/gliner2-large-v1` | 340M | Local | Higher-capacity local inference when latency and memory allow |
| `fastino/gliner2-multi-v1` | about 300M | Local | Multilingual inference and specialization; evaluate each target language |
| GLiNER XL | 1B | Hosted API | Hosted extraction without local model weights |

The base model card is English-tagged, while the multilingual model is the intended choice beyond the base/large language coverage. Fastino describes `gliner2-multi-v1` as broadly multilingual but does not publish task-level quality for every language. Do not infer validated support from tokenizer acceptance or tutorial examples. Read [references/concepts-and-models.md](references/concepts-and-models.md) and evaluate every target language on representative labeled data.

## Install with uv

Use the verified source revision so local inference, long-context helpers, and
training APIs match this skill. In an existing uv-managed project:

```bash
uv add "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"
```

For an isolated experiment:

```bash
uv venv .venv-gliner2 --python 3.12
uv pip install --python .venv-gliner2/bin/python \
  "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"
.venv-gliner2/bin/python your_script.py
```

The full commit is intentional: update it only after validating the replacement
revision and all bundled scripts. Do not install into an ambiguous system
interpreter. If installation, version, source, or capability checks disagree,
use [references/troubleshooting.md](references/troubleshooting.md).

## Route the task

| Task | Read | Start from |
|---|---|---|
| Choose a model, device, execution mode, or language strategy | [references/concepts-and-models.md](references/concepts-and-models.md) | — |
| Run NER, classification, relations, combined, batch, API, or long-document inference | [references/inference.md](references/inference.md) | [scripts/infer_entities.py](scripts/infer_entities.py) |
| Build structures or interpret output shapes | [references/schemas-and-outputs.md](references/schemas-and-outputs.md) | [scripts/infer_structured.py](scripts/infer_structured.py) |
| Prepare or validate `InputExample`/JSONL training data | [references/training-data.md](references/training-data.md) | [scripts/validate_training_data.py](scripts/validate_training_data.py) |
| Perform full fine-tuning | [references/fine-tuning.md](references/fine-tuning.md) | [scripts/train_gliner2.py](scripts/train_gliner2.py) |
| Train, load, save, or merge LoRA adapters | [references/lora-and-adapters.md](references/lora-and-adapters.md) | [scripts/train_gliner2_lora.py](scripts/train_gliner2_lora.py) |
| Compare a base and specialized model | [references/fine-tuning.md](references/fine-tuning.md) | [scripts/evaluate_gliner2.py](scripts/evaluate_gliner2.py) |
| Diagnose a failing consumer workflow | [references/troubleshooting.md](references/troubleshooting.md) | — |

Read each selected reference completely before adapting its script. Copy the bundled script as the starting point instead of recreating a training or inference program from memory.
Command examples beginning with `scripts/` assume the current directory is this
skill directory. Otherwise use the script's resolved absolute path or copy the
script into the consumer project before running it.

## Recognize named entities

```python
from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
text = "Apple CEO Tim Cook introduced Vision Pro in Cupertino."

result = model.extract_entities(
    text,
    {
        "company": "Company or organization names",
        "person": "Names of people",
        "product": "Commercial product names",
        "location": "Cities or physical locations",
    },
    include_confidence=True,
    include_spans=True,
)

for items in result["entities"].values():
    for item in items:
        assert text[item["start"]:item["end"]] == item["text"]
```

Descriptions disambiguate domain labels. Tune thresholds on a held-out evaluation set, not the example used to demonstrate the schema.

## Extract structured records

```python
from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
text = "Invoice INV-42 from Northwind. Status: unpaid. One keyboard costs USD 75."
schema = (
    model.create_schema()
    .structure("invoice")
        .field("invoice_number", dtype="str")
        .field("vendor", dtype="str")
        .field("status", dtype="str", choices=["paid", "unpaid", "partial", "overdue"])
    .structure("line_item")
        .field("description", dtype="str")
        .field("quantity", dtype="str")
        .field("unit_price", dtype="str")
        .field("total", dtype="str")
)

built = schema.build()  # model-free schema inspection is also available through Schema()
result = model.extract(text, schema, include_confidence=True, include_spans=True)
```

Structures are flat named records. Repeated line items are sibling `line_item` instances, not children nested inside `invoice`. Split multi-parent documents or associate siblings deterministically downstream.

## Preserve GLiNER2 contracts

- Distinguish quick-API dictionaries from schema-builder calls. Pass builder classification options such as `multi_label` and `cls_threshold` as keyword arguments.
- Request `include_spans=True` whenever code consumes `start` or `end`; request `include_confidence=True` independently.
- Expect choice fields such as a structured status to omit character spans.
- Treat `text[start:end] == item["text"]` as offset integrity, not semantic correctness.
- Keep relation descriptions directional and validate expected head/tail order.
- Keep structures flat; do not invent nested builders, `structure(..., dtype=...)`, or `end_structure()`.
- For repeated structures, validate expected record cardinality and within-record field coherence. Correct spans do not prove that fields from different clauses or events belong to the same sibling record.
- Prefer a field-specific threshold when one structured field needs more recall. Lowering a global threshold can increase false positives across every task.
- Detect missing and `null` required fields before deterministic parsing. Never turn a missing amount into zero or a missing identifier into a placeholder.
- Validate the actual object and imported package path before using less-common, hosted-only, or local-only methods.

## Choose full fine-tuning or LoRA

Improve label descriptions, schema design, thresholds, preprocessing, and deterministic validation before training. Fine-tune only when representative evaluation shows a repeatable domain gap.

Choose full fine-tuning when maximum adaptation justifies updating and storing the complete model. Choose LoRA when smaller trainable state, lower memory use, adapter reuse, or multiple domain specializations matter. In both cases:

1. Validate and split data before loading the model.
2. Establish a base-model evaluation result.
3. Run a one-step smoke test before a long job.
4. Evaluate on held-out data with task-appropriate metrics.
5. Reload the saved model or adapter into a fresh base model and repeat a bounded inference test.

Example full-training entry point:

```bash
uv run --script scripts/train_gliner2.py \
  --output-dir outputs/gliner2-domain \
  --smoke-test
```

Use [scripts/train_gliner2_lora.py](scripts/train_gliner2_lora.py) for adapter training. Keep adapter-only artifacts distinct from merged or fully fine-tuned models, and load new adapters through PEFT-native APIs described in the LoRA reference.

## Validate the result

For inference, report API/implementation validity separately from semantic extraction quality. For training, record the resolved model, package path/version, device, precision, dataset counts, seed, config, baseline metrics, final metrics, output path, and reload result.

Do not award a task-level pass merely because code executed. Required fields, expected labels or relations, numeric checks, artifact reload, and explicit scenario requirements must also pass. Use [references/troubleshooting.md](references/troubleshooting.md) when output shape, package capability, batching, retry, threshold, adapter, or environment behavior differs from expectation.
