# GLiNER2 troubleshooting

Use this reference to diagnose consumer-side inference, schema, API, data, training, and adapter failures. Start from observable package paths, public outputs, and exact capabilities. Do not repair a native-capability test by monkeypatching GLiNER2.

## Contents

- [Resolve the environment first](#resolve-the-environment-first)
- [Diagnose package/source capability mismatches](#diagnose-packagesource-capability-mismatches)
- [Diagnose model loading and revision problems](#diagnose-model-loading-and-revision-problems)
- [Fix schema and output-shape mistakes](#fix-schema-and-output-shape-mistakes)
- [Fix repeated-record undercount and field mixing](#fix-repeated-record-undercount-and-field-mixing)
- [Handle empty, null, or low-confidence output](#handle-empty-null-or-low-confidence-output)
- [Validate spans, overlap, and long documents](#validate-spans-overlap-and-long-documents)
- [Prove real batching](#prove-real-batching)
- [Test the hosted API correctly](#test-the-hosted-api-correctly)
- [Fix device and precision failures](#fix-device-and-precision-failures)
- [Fix training-data and evaluation failures](#fix-training-data-and-evaluation-failures)
- [Fix LoRA and adapter failures](#fix-lora-and-adapter-failures)
- [Report a reproducible result](#report-a-reproducible-result)

## Resolve the environment first

Use uv with a dedicated interpreter. Do not alternate between a system Python, an activated environment, and `uv run` without proving they resolve to the same executable.

```bash
uv venv .venv-gliner2 --python 3.12
uv pip install \
  --python .venv-gliner2/bin/python \
  "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"
.venv-gliner2/bin/python - <<'PY'
import inspect
import sys
import gliner2
from gliner2 import GLiNER2

print("python:", sys.executable)
print("gliner2 path:", gliner2.__file__)
print("gliner2 version:", gliner2.__version__)
print("from_pretrained:", inspect.signature(GLiNER2.from_pretrained))
PY
```

Record `uv --version`, the exact install command, `sys.executable`, package path, package version, and device. If an editable checkout is intended, the printed package path must point into that checkout. A matching version string does not prove matching source.

## Diagnose package/source capability mismatches

The PyPI `1.3.2` wheel predates repository long-context additions, while later repository source still reports `1.3.2`. Therefore version comparison alone cannot answer whether native long methods exist.

Probe the class from the environment that will execute inference:

```bash
.venv-gliner2/bin/python - <<'PY'
import gliner2
from gliner2 import GLiNER2

print(gliner2.__version__, gliner2.__file__)
for name in (
    "extract_long",
    "batch_extract_long",
    "extract_entities_long",
    "batch_extract_entities_long",
):
    print(name, hasattr(GLiNER2, name))
PY
```

If the methods are required but absent, the executing environment did not use
the skill's pinned source baseline. Recreate it in an isolated uv environment:

```bash
uv venv .venv-gliner2-source --python 3.12
uv pip install \
  --python .venv-gliner2-source/bin/python \
  "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2.git@31c8abaa4a6d88ae8bb6f2e63cfea9926956497c"
```

Re-run the probe with `.venv-gliner2-source/bin/python`. Do not attach methods dynamically or call application chunking “native GLiNER2.” Keep any fallback result separate from the native capability result.

## Diagnose model loading and revision problems

Model downloads and GLiNER2 package installation are separate. Local inference with a public Hugging Face model normally needs no GLiNER2 API key, but it needs network access on first download unless the files are cached.

In the current source, `GLiNER2.from_pretrained(repo_or_dir, **kwargs)` consumes `quantize`, `compile`, and `map_location`; it does not forward a `revision=` value to each Hugging Face download. Passing `revision=` therefore does not pin all loaded artifacts. For reproducibility, resolve an approved model snapshot ahead of time, verify its files/hash, and load its local directory. Do not claim revision pinning merely because the keyword was accepted.

When loading fails, distinguish:

- package import failure;
- model repository/network/authentication failure;
- missing `config.json`, `encoder_config/config.json`, tokenizer, or weights in a local snapshot;
- base-model/adapter mismatch;
- memory or dtype failure after successful download.

## Fix schema and output-shape mistakes

The quick APIs and builder API accept different shapes. For quick multi-label classification, use the wrapper:

```python
model.classify_text(text, {
    "topics": {
        "labels": ["technology", "business"],
        "multi_label": True,
        "cls_threshold": 0.3,
    }
})
```

For the builder, pass options as keywords:

```python
schema = model.create_schema().classification(
    "topics",
    ["technology", "business"],
    multi_label=True,
    cls_threshold=0.3,
)
```

A dictionary passed as builder labels means `{label: description}`; it is not the quick-API wrapper.

Structures are flat sibling record types:

```python
schema = (
    model.create_schema()
    .structure("invoice")
        .field("invoice_number", dtype="str")
    .structure("line_item")
        .field("description", dtype="str")
        .field("total", dtype="str")
)
```

Do not use nested structures, `structure(..., dtype=...)`, or `end_structure()`. When several parent records occur in one document, split the input or associate siblings downstream with deterministic keys.

Request output metadata explicitly and independently:

- `include_spans=True` for `start` and `end`;
- `include_confidence=True` for confidence;
- neither flag implies the other.

Choice/classification values can be predictions without a source span. Do not reject them solely because they lack `start`/`end`.

Inspect built schemas through public `build()` or `to_dict()` output where available. Do not build application assertions against `_field_metadata` or another underscore-prefixed implementation detail.

## Fix repeated-record undercount and field mixing

Repeated sibling structures can fail in two distinct ways:

- **undercount:** the text describes several events but the model returns fewer records;
- **field mixing:** one record combines individually valid spans from different subjects, clauses, or events.

Span equality and required-field presence do not detect either failure. Diagnose
and recover in this order:

1. Preserve and inspect the raw result before filtering or normalization.
2. Compare returned structure count with any expected event count established by
   the input, an upstream parser, or labeled evaluation data.
3. Check that each record's fields share the same subject, clause, event, time,
   or explicit grouping identifier.
4. Make the structure and field descriptions more specific about event
   boundaries and field roles.
5. Run the structure as a separate extraction task when a combined schema
   undercounts or cross-binds fields.
6. Tune only relevant field thresholds on held-out data; do not lower the global
   threshold to force a missing sibling into one demonstration.
7. Fine-tune only when representative labeled evaluation shows the failure is
   persistent after schema and execution changes.

For negation and contrast, keep raw model output separate from deterministic
post-processing. Use scope-aware, domain-tested rules rather than deleting
topics or records whenever one positive or negative keyword appears. Always
report raw semantic quality as well as final application-output quality.

## Handle empty, null, or low-confidence output

An existing task mapping does not prove extraction success. Check required lists and fields recursively:

- `{entities: {person: []}}` contains no person;
- a structure with `problem_description: null` is incomplete;
- a relation mapping with no tuples is empty;
- a missing amount must not become `0`;
- a missing identifier must not become a placeholder.

Separate three questions:

1. Did the API and schema execute correctly?
2. Are spans and output shapes structurally valid?
3. Are the predicted values semantically correct for the task?

A structurally valid run can still be semantically partial or failed.

If one field needs more recall, assign a field threshold in the builder instead of lowering the global extraction threshold for every task. Treat a value captured only at a low confidence as requiring review, especially when its boundary is incomplete (for example `60` instead of `USD 60`). Tune thresholds on held-out data, not on the demonstration input.

Mandatory human review is a business rule, not merely an empty-output fallback. A medical, legal, or other high-impact workflow may require review even when entities were found. Encode `requires_review` from the policy and risk signals, including missing/contradictory fields and low confidence.

## Validate spans, overlap, and long documents

For every returned item carrying offsets, check half-open spans:

```python
assert 0 <= item["start"] <= item["end"] <= len(text)
assert text[item["start"]:item["end"]] == item["text"]
```

Traverse nested structures and relation endpoints recursively. This proves offset integrity, not correct labels or complete boundaries.

Define overlap policy separately from exact deduplication. Exact duplicate removal does not resolve nested or partially overlapping spans. Decide whether the application keeps nested mentions, applies non-maximum suppression, or flags collisions for review; then test that policy on boundary cases.

For native long-document methods:

- prove the method exists on the imported class;
- show its signature;
- force input to span multiple chunks;
- verify returned offsets against the complete original text;
- distinguish exact deduplication from overlap resolution;
- report semantic quality independently from chunk/offset correctness.

Do not replace a missing method with a custom chunker during a native test. A custom tokenizer that counts whitespace tokens or merges only exact duplicates has different semantics and must be evaluated as a separate application implementation.

## Prove real batching

Using a function named `batch_*` or accepting `batch_size` is insufficient. A wrapper such as this is sequential:

```python
def not_real_batch(texts, batch_size=8):
    return [model.extract(text, schema) for text in texts]
```

For multiple documents, call the native GLiNER2 batch method and verify:

- one output per input;
- input/output ordering;
- `batch_size` reaches native batch inference;
- changing it alters grouping without altering ordering;
- no hidden sequential fallback replaced the implementation.

Report “sequential processing” when that is what ran. Do not count it as a batching pass.

## Test the hosted API correctly

`GLiNER2.from_api()` reads `PIONEER_API_KEY` unless `api_key=` is supplied. Test the missing-key branch without setting either source. Never print headers, environment secrets, or full exception payloads that can contain credentials.

The client distinguishes authentication, validation, server, and general API errors. Its `requests.Session` mounts an `HTTPAdapter` with urllib3 retries for POST on `429`, `500`, `502`, `503`, and `504`, using backoff.

Replacing `client.session.post` with a lambda bypasses the mounted adapter and therefore does not test retry behavior. For retry assertions, exercise the adapter/transport boundary with a controlled local server or mock the adapter at the appropriate level. Assert call count and backoff/retry status selection. Test non-retryable responses separately.

For offline tests, fail on every unexpected network call; do not merely mock one expected endpoint. Verify that secrets are absent from captured logs and raised messages.

Local and hosted extractors are not capability-identical. Probe optional methods on the actual object before designing a runtime switch.

## Fix device and precision failures

Verify the device from model parameters, not from requested configuration:

```python
devices = sorted({str(p.device) for p in model.parameters()})
print(devices)
```

Current `GLiNER2Trainer` automatically selects CUDA when available and otherwise CPU; it does not select MPS. Its setup disables fp16/bf16 on CPU. A consumer template should therefore:

- use fp16 only on supported CUDA;
- use bf16 only when `torch.cuda.is_bf16_supported()`;
- use fp32 on CPU;
- reject a requested device that the trainer would silently ignore;
- report that MPS inference may work while this trainer still falls back to CPU.

Do not enable fp16 merely because a tutorial configuration did. For CUDA out-of-memory failures, reduce batch size, sequence length, LoRA rank/targets, or increase gradient accumulation while preserving the intended effective batch size.

## Fix training-data and evaluation failures

Load and strictly validate before model download:

```python
from gliner2.training.data import TrainingDataset

dataset = TrainingDataset.load("train.jsonl")
report = dataset.validate(raise_on_error=True)
relation_errors = dataset.validate_relation_consistency()
if relation_errors:
    raise ValueError("\n".join(relation_errors))
```

Reject empty datasets and invalid references. Entity mentions, structure field values, and relation values must occur in the source text. Keep a relation type's field names consistent across examples. Split before training and prevent duplicate or near-duplicate document leakage.

`TrainingConfig` defaults to `eval_strategy="steps"` and `fp16=True`. Resolve both explicitly:

- without eval data: set `eval_strategy="no"`, `save_best=False`, and disable early stopping;
- with eval data: choose `steps` or `epoch` deliberately;
- early stopping requires non-empty eval data;
- CPU requires fp16/bf16 off.

The trainer's validation path may sanitize invalid records. A reusable training script should run `TrainingDataset.validate(raise_on_error=True)` first so malformed data fails instead of silently shrinking the job. After training, reload the actual `final` or existing `best` checkpoint and run a bounded inference/evaluation check.

## Fix LoRA and adapter failures

Use the PEFT-native path described in [lora-and-adapters.md](lora-and-adapters.md):

```python
base = GLiNER2.from_pretrained(base_model_id)
adapted = PeftModel.from_pretrained(base, adapter_dir)
```

Diagnose adapters in this order:

1. Confirm `adapter_config.json` exists and contains `peft_type: "LORA"`.
2. Confirm `adapter_model.safetensors` or `adapter_model.bin` exists.
3. Inspect `base_model_name_or_path` and load the matching base/snapshot.
4. Compare target module names with the base architecture.
5. Confirm LoRA parameter names exist and were trainable during training.
6. Reload in a fresh process and compare adapter parameter names, shapes, dtypes, and values.
7. Repeat held-out evaluation.

Do not write the legacy GLiNER2 adapter shape and PEFT-native shape into the same directory. The legacy writer can replace PEFT's `adapter_config.json`, remove `peft_type`, and cause `PeftModel.from_pretrained()` to fail.

New code should not call deprecated `model.load_adapter`, `model.save_adapter`, `model.unload_adapter`, `model.merge_lora`, or the legacy functions in `gliner2.training.lora`. Use `apply_lora`, `PeftModel.save_pretrained`, `PeftModel.from_pretrained`, named PEFT adapters, and `merge_and_unload`.

If no parameters are trainable, confirm `use_lora=True` and that target groups resolve to actual linear modules. If an adapter appears to have no effect, compare base and adapted predictions/metrics and inspect active adapter state; identical output for one example is not itself proof of a loading failure.

Treat adapter-only and merged artifacts differently:

- adapter-only: load with a fresh matching base plus `PeftModel.from_pretrained`;
- merged/full model: load directly with `GLiNER2.from_pretrained`;
- never pass an adapter-only directory to `GLiNER2.from_pretrained` as though it contained full weights.

## Report a reproducible result

For inference, record:

- uv and Python versions plus `sys.executable`;
- GLiNER2 version and `gliner2.__file__`;
- exact package/source install command and source commit when relevant;
- model identifier or verified local snapshot;
- actual parameter device and dtype;
- schema, thresholds, flags, and method signatures;
- raw concise output and assertions;
- API/structural validity separately from semantic quality.

For training, additionally record dataset counts/splits, validation report, seed, resolved `TrainingConfig`, LoRA targets/counts, checkpoint path, base identity, and fresh reload/evaluation result.

Assign one scenario status. `PASS` requires every explicit behavior and assertion; a missing required field is at best `PARTIAL`, even if the script exited successfully. Do not double-count subchecks as additional scenarios or call a custom fallback a native result.
