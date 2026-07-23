# LoRA and PEFT adapters

Use this reference when specializing GLiNER2 with LoRA, loading an adapter, merging one for deployment, or serving several domain adapters. The current public path is `Extractor.apply_lora()` plus PEFT-native save/load APIs. The older GLiNER2 adapter methods remain only as compatibility shims.

## Contents

- [Choose LoRA or full fine-tuning](#choose-lora-or-full-fine-tuning)
- [Understand the GLiNER2 LoRA controls](#understand-the-gliner2-lora-controls)
- [Train with the bundled template](#train-with-the-bundled-template)
- [Inspect trainable parameters](#inspect-trainable-parameters)
- [Save and reload an adapter](#save-and-reload-an-adapter)
- [Merge for standalone deployment](#merge-for-standalone-deployment)
- [Use multiple domain adapters](#use-multiple-domain-adapters)
- [Evaluate and publish](#evaluate-and-publish)
- [Avoid the legacy adapter API](#avoid-the-legacy-adapter-api)

## Choose LoRA or full fine-tuning

LoRA freezes the base GLiNER2 weights and learns low-rank updates for selected linear layers. It is a good default when GPU memory, training time, storage, or maintaining several domain variants matters. Its saved artifact contains adapter configuration and deltas, not a complete GLiNER2 model; inference therefore needs the same base model.

Use full fine-tuning when the evaluation gain justifies updating and distributing the complete model, or when LoRA cannot close the measured domain gap. Do not choose either approach before establishing a held-out base-model result. Schema descriptions, label design, thresholds, and deterministic validation often fix problems without training.

Compare both approaches on the same data split and metrics. Parameter-efficiency does not imply higher accuracy, and tutorial estimates for adapter size or training speed are not guarantees for a particular model, target set, device, or dataset.

## Understand the GLiNER2 LoRA controls

`TrainingConfig` exposes:

```python
use_lora=True
lora_r=8
lora_alpha=16.0
lora_dropout=0.0
lora_use_dora=False
lora_target_modules=["encoder"]
save_adapter_only=True
```

- `lora_r` is the low-rank bottleneck capacity. A larger rank trains and stores more parameters.
- `lora_alpha` scales the update by approximately `alpha / r`; `2 * r` is a starting convention, not an optimization result.
- `lora_dropout` regularizes the adapter branch. Tune it on validation data.
- `lora_use_dora` asks PEFT to use DoRA weight decomposition.
- `save_adapter_only=True` makes trainer checkpoints PEFT-native adapter directories.

GLiNER2 resolves these high-level target groups against linear modules:

| Target | Resolved area |
|---|---|
| `encoder` | Encoder linear layers named like query, key, value, or dense |
| `encoder.query` | Encoder query projections |
| `encoder.key` | Encoder key projections |
| `encoder.value` | Encoder value projections |
| `encoder.dense` | Encoder dense projections |
| `span_rep` | Span-representation linear layers |
| `classifier` | Classifier linear layers |
| `count_embed` | Count-embedding linear layers |
| `count_pred` | Count-prediction linear layers |

Start with `encoder` for a lower-capacity adapter. Add task heads only when held-out results show that encoder-only adaptation is insufficient. `TrainingConfig` itself defaults to all supported groups, which trains a larger adapter than the bundled script's safer `encoder` default.

Call the model API directly only when writing a custom trainer:

```python
from gliner2 import GLiNER2

base = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
peft_model = base.apply_lora(
    r=8,
    alpha=16.0,
    dropout=0.0,
    targets=["encoder"],
)
```

`apply_lora()` returns a `peft.PeftModel`; retain that returned object. The current `GLiNER2Trainer` applies this operation automatically when `TrainingConfig(use_lora=True)` is used.

## Train with the bundled template

Use [../scripts/train_gliner2_lora.py](../scripts/train_gliner2_lora.py) rather than recreating a training program. It normalizes both supported JSONL forms, strictly validates data, rejects train/eval text overlap and cross-split schema drift, resolves precision against the trainer's actual device behavior even in `--validate-only` mode, records parameter counts, saves adapter-only checkpoints, reloads `final` into a fresh base model, and repeats a bounded inference probe in a clean Python process.

Run a model-free validation first:

```bash
uv run --script scripts/train_gliner2_lora.py \
  --train-data train.jsonl \
  --eval-data validation.jsonl \
  --output-dir outputs/legal-lora \
  --validate-only
```

Then run one bounded optimization step:

```bash
uv run --script scripts/train_gliner2_lora.py \
  --train-data train.jsonl \
  --eval-data validation.jsonl \
  --output-dir outputs/legal-lora-smoke \
  --smoke-test
```

Use a new output directory for the full run:

```bash
uv run --script scripts/train_gliner2_lora.py \
  --train-data train.jsonl \
  --eval-data validation.jsonl \
  --output-dir outputs/legal-lora \
  --epochs 5 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-target encoder
```

Without `--eval-data`, the template sets `eval_strategy="no"` and disables best-checkpoint selection. With evaluation data, it evaluates at configured steps and may also produce `best`. The always-produced adapter for a completed run is `OUTPUT_DIR/final`.

The script does not push to a Hub and does not accept an implicit production dataset. `--smoke-test` is the only mode that supplies small synthetic data; it also forces `max_steps=1`.

## Inspect trainable parameters

Count parameters after the trainer has applied LoRA:

```python
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())
percentage = 100.0 * trainable / total
print(trainable, total, percentage)
```

Inspect names when verifying the freeze boundary:

```python
names = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
assert names
assert all("lora_" in name for name in names)
```

Record the count and target groups with every run. Never copy parameter counts from a tutorial: they depend on the selected GLiNER2 model, PEFT version, rank, DoRA choice, and resolved target modules.

## Save and reload an adapter

When `use_lora=True` and `save_adapter_only=True`, `GLiNER2Trainer` calls PEFT's `save_pretrained()` on its `PeftModel`. A valid current checkpoint contains at least:

```text
adapter_config.json
adapter_model.safetensors   # or adapter_model.bin
```

`adapter_config.json` must remain the PEFT config and contain `peft_type: "LORA"`. It also records `base_model_name_or_path`; keep this identity alongside dataset, code, package, and model revisions.

Reload against a fresh matching base:

```python
from gliner2 import GLiNER2
from peft import PeftModel

base = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
adapted = PeftModel.from_pretrained(base, "outputs/legal-lora/final")
adapted.eval()

result = adapted.extract_entities(
    "Northwind filed a claim against Contoso.",
    ["company", "legal_action"],
)
```

A successful file load is not a complete round-trip test. Verify:

1. `base_model_name_or_path` is present and matches the intended base.
2. Adapter parameter names, shapes, dtypes, and values match the just-saved model.
3. A fresh-process inference call succeeds on a representative example.
4. Held-out metrics match the pre-save adapter within the expected numerical tolerance.

## Merge for standalone deployment

Merge only when the consumer needs one standalone specialized model and no longer needs adapter switching:

```python
from gliner2 import GLiNER2
from peft import PeftModel

base = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
adapted = PeftModel.from_pretrained(base, "outputs/legal-lora/final")
merged = adapted.merge_and_unload()
merged.save_pretrained("outputs/legal-merged")
```

The merged directory is a complete GLiNER2 model, not an adapter. Reload it with `GLiNER2.from_pretrained()`. Evaluate the merged artifact separately because dtype conversion and merge behavior can change outputs slightly. Preserve the original adapter and its metadata until merged-model validation is complete.

## Use multiple domain adapters

Train each domain into a distinct immutable adapter directory against the same recorded base revision. PEFT can keep named adapters on one wrapped model:

```python
from gliner2 import GLiNER2
from peft import PeftModel

base = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
model = PeftModel.from_pretrained(base, "adapters/legal", adapter_name="legal")
model.load_adapter("adapters/medical", adapter_name="medical")

model.set_adapter("legal")
legal = model.extract_entities("Contoso filed suit.", ["company", "legal_action"])

model.set_adapter("medical")
medical = model.extract_entities("Metformin 500 mg was prescribed.", ["drug", "dosage"])
```

Group batches by adapter to avoid switching for every document. Do not mutate the active adapter concurrently from multiple requests; use synchronization or separate model workers. Validate that every adapter declares the same intended base before attaching it to a shared instance. Keep domain routing outside the extraction model and evaluate wrong-route behavior explicitly.

## Evaluate and publish

Compare base, active adapter, freshly reloaded adapter, and optional merged model on the same held-out set. Use exact span/label metrics for entities, directional tuples for relations, task-appropriate classification metrics, and required-field coverage plus normalized values for structures.

Publish only after preserving:

- exact base model identifier or verified local snapshot;
- GLiNER2, PEFT, PyTorch, and Python versions;
- LoRA rank, alpha, dropout, DoRA, and target groups;
- train/eval split identities and validation report;
- trainable and total parameter counts;
- adapter config and output hash;
- fresh-base reload and held-out evaluation results.

The bundled script intentionally has no automatic Hub push. Uploading is a separate, explicit deployment action.

## Avoid the legacy adapter API

Do not copy `model.load_adapter()`, `model.unload_adapter()`, `model.save_adapter()`, `model.merge_lora()`, or `save_pretrained(save_adapter_only=True)` from Tutorials 10–11 into new code. They emit `PendingDeprecationWarning` and use compatibility behavior.

Also avoid the compatibility functions in `gliner2.training.lora`, including `apply_lora_to_model`, `save_lora_adapter`, `load_lora_adapter`, `merge_lora_weights`, and `remove_lora_from_model`. In particular, the legacy adapter directory uses `adapter_weights.safetensors` and a GLiNER2-specific config, while current PEFT uses `adapter_model.safetensors` and a config containing `peft_type`. Mixing writers in one directory can clobber `adapter_config.json` and make `PeftModel.from_pretrained()` fail.

Use this mapping for new code:

| Legacy operation | Current operation |
|---|---|
| Apply LoRA | `base.apply_lora(...)` or `TrainingConfig(use_lora=True)` |
| Save adapter | `PeftModel.save_pretrained(path)` |
| Load adapter | `PeftModel.from_pretrained(fresh_base, path)` |
| Add another adapter | `PeftModel.load_adapter(path, adapter_name=...)` |
| Select adapter | `PeftModel.set_adapter(name)` |
| Merge permanently | `PeftModel.merge_and_unload()` |

Use [troubleshooting.md](troubleshooting.md) for adapter/base mismatches, missing PEFT files, precision failures, empty trainable sets, and reload discrepancies.
