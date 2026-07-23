# Full fine-tuning GLiNER2

Use this reference for full-parameter domain adaptation. Read `training-data.md` first and use `lora-and-adapters.md` instead when adapter-only training is the better tradeoff.

## Contents

- [When to fine-tune](#when-to-fine-tune)
- [Verified training API](#verified-training-api)
- [Configuration decisions](#configuration-decisions)
- [Device and precision](#device-and-precision)
- [Evaluation, early stopping, and checkpoints](#evaluation-early-stopping-and-checkpoints)
- [Smoke-to-full workflow](#smoke-to-full-workflow)
- [Bundled training template](#bundled-training-template)
- [Task-aware evaluation](#task-aware-evaluation)
- [Operational checklist](#operational-checklist)

## When to fine-tune

Fine-tune after testing schema wording, descriptions, field cardinality, thresholds, and a stronger base checkpoint. Fine-tuning is appropriate when a representative labeled validation set shows a persistent domain, terminology, or task-definition gap.

Keep a frozen base-model baseline. A falling training loss is not proof that entity, relation, classification, or structure quality improved. Compare task metrics on held-out data and inspect error slices.

Choose full fine-tuning when maximum adaptation matters and memory permits updating the encoder and task heads. Choose LoRA for smaller artifacts, lower memory, or several swappable domains. Do not mix full-model checkpoints and adapter-only directories.

## Verified training API

The current public entry points are:

```python
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./runs/domain-v1",
    num_epochs=3,
    batch_size=2,
    gradient_accumulation_steps=4,
    encoder_lr=1e-5,
    task_lr=5e-4,
    fp16=False,
    bf16=False,
    eval_strategy="epoch",
    save_best=True,
)
trainer = GLiNER2Trainer(model=model, config=config)
results = trainer.train(train_data="train.jsonl", eval_data="validation.jsonl")
```

The keyword is `eval_data`, not `val_data`. Training accepts a JSONL path, a list of paths, a list of `InputExample`, a `TrainingDataset`, or raw record dictionaries.

Current `TrainingConfig` supports these full-training controls:

| Area | Fields |
|---|---|
| Length | `num_epochs`, `max_steps` |
| Batch | `batch_size`, `eval_batch_size`, `gradient_accumulation_steps` |
| Optimizer | `encoder_lr`, `task_lr`, `weight_decay`, Adam betas/epsilon, `max_grad_norm` |
| Schedule | `scheduler_type`, `warmup_ratio`, `warmup_steps`, `num_cycles` |
| Precision | `fp16`, `bf16` |
| Evaluation/save | `eval_strategy`, `eval_steps`, `save_total_limit`, `save_best`, `metric_for_best`, `greater_is_better` |
| Early stop | `early_stopping`, patience, threshold |
| Runtime | `num_workers`, `pin_memory`, `seed`, `deterministic`, `local_rank`, `max_len` |
| Sampling | `max_train_samples`, `max_eval_samples` |

The tutorial's `gradient_checkpointing` example is stale: it is not a field in the current `TrainingConfig`. Do not pass it unless the installed signature proves support.

## Configuration decisions

### Separate learning rates

Full fine-tuning places parameters whose names contain `encoder` in an optimizer group using `encoder_lr`; other trainable task parameters use `task_lr`. Start conservatively, commonly around `1e-5` for the encoder and `5e-4` for task-specific parameters, then tune against validation metrics rather than copying fixed values.

### Effective batch size

For one process:

```text
effective batch = batch_size × gradient_accumulation_steps
```

Gradient accumulation divides each batch loss and updates after the requested number of batches. The trainer flushes an incomplete accumulation cycle at epoch end. Reduce the micro-batch first when memory is limited, then increase accumulation to recover the intended effective batch.

### Length

`max_len` controls word-token truncation in the training and evaluation collator. Any annotation beyond the truncated region cannot contribute correctly. Measure corpus lengths before choosing it and include long-document edge cases in evaluation.

### Scheduler and warmup

Supported scheduler names are `linear`, `cosine`, `cosine_restarts`, and `constant`. A positive `warmup_steps` overrides the ratio-derived warmup. `max_steps > 0` overrides epoch-derived optimization steps and is the reliable bound for smoke tests.

### Reproducibility

Set and record `seed`. `deterministic=True` configures deterministic cuDNN behavior where applicable, but complete bitwise reproducibility still depends on PyTorch, kernels, drivers, workers, and hardware. Record package path/version, Python, Torch/CUDA, model identifier or local checkpoint, resolved config, data hashes, and actual parameter devices.

## Device and precision

Current `GLiNER2Trainer` selects CUDA when available and otherwise CPU. It does not select Apple MPS, even if MPS inference works. Treat an MPS-only machine as CPU training unless the trainer implementation changes and is verified.

The library defaults `fp16=True`, which is unsafe to copy blindly. The trainer disables mixed precision on CPU. Prefer explicit settings:

| Runtime | Recommended starting point |
|---|---|
| CUDA with standard support | `fp16=True`, `bf16=False` |
| CUDA with verified BF16 support | `fp16=False`, `bf16=True` |
| CPU or current MPS-only path | both `False` |

Never enable FP16 and BF16 together. The bundled template resolves `--precision auto` to FP16 only on CUDA and FP32 otherwise, and rejects unsupported explicit mixed precision.

## Evaluation, early stopping, and checkpoints

Choose `eval_strategy` deliberately:

- `steps`: evaluate at `eval_steps` intervals and save checkpoints at those steps.
- `epoch`: evaluate and save at each epoch end.
- `no`: skip periodic evaluation; the trainer still saves `final`.

Use `eval_strategy="no"` when no evaluation dataset is supplied. `save_best=True` is meaningful only when evaluation runs. Early stopping requires non-empty `eval_data` and an active evaluation strategy.

The trainer's built-in evaluation primarily reports loss components. Supply task-aware held-out evaluation separately; do not select a domain model solely from training loss.

Checkpoint layout is under `output_dir`:

```text
output_dir/
├── training_config.json
├── checkpoint-.../       # step strategy
├── checkpoint-epoch-.../ # epoch strategy
├── best/                 # only after an improving evaluation when save_best=True
└── final/                # always saved at normal training completion
```

Reload a full checkpoint with:

```python
from gliner2 import GLiNER2

reloaded = GLiNER2.from_pretrained("./runs/domain-v1/final", map_location="cpu")
```

`GLiNER2Trainer.load_checkpoint()` restores model weights, not optimizer, scheduler, scaler, epoch, or global-step state. Continuing after it is a weights-only warm start, not an exact resume. The bundled script calls this option `--warm-start-checkpoint` to avoid overstating reproducibility.

## Smoke-to-full workflow

Use three stages:

1. **Model-free preflight:** validate JSONL, schema consistency, split overlap, counts, and configuration.
2. **One-step smoke:** load the intended model, run exactly one optimizer step, save `final`, reload it, and perform one inference probe.
3. **Bounded experiment:** use a small but representative subset, active validation, and task-aware before/after evaluation.
4. **Full run:** only after the bounded experiment has correct artifacts and improves the target metrics.

The one-step smoke establishes plumbing, not model quality. Do not compare its semantic output as if it were a trained model.

## Bundled training template

Run the portable script with `uv`; its PEP 723 block declares dependencies:

```bash
uv run /path/to/gliner2/scripts/train_gliner2.py \
  --smoke-test \
  --output-dir ./runs/smoke-full
```

Smoke mode uses two synthetic NER examples and forces `max_steps=1`. It refuses simultaneous external data paths.

Start a bounded real run:

```bash
uv run /path/to/gliner2/scripts/train_gliner2.py \
  --train-data data/train.jsonl \
  --eval-data data/validation.jsonl \
  --output-dir runs/domain-v1-bounded \
  --model fastino/gliner2-base-v1 \
  --max-steps 100 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --precision auto \
  --eval-strategy steps \
  --eval-steps 25
```

Then launch an epoch-based run only after reviewing the bounded artifacts:

```bash
uv run /path/to/gliner2/scripts/train_gliner2.py \
  --train-data data/train.jsonl \
  --eval-data data/validation.jsonl \
  --output-dir runs/domain-v1 \
  --num-epochs 10 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --encoder-lr 1e-5 \
  --task-lr 5e-4 \
  --precision auto \
  --eval-strategy epoch \
  --early-stopping \
  --early-stopping-patience 3
```

The script is safe by default:

- requires explicit train data unless smoke mode is used;
- validates before loading the model;
- rejects exact train/eval text overlap by default;
- refuses a non-empty output directory unless explicitly allowed;
- never pushes to the Hub;
- prints environment, package path, resolved precision, devices, and config;
- writes `run_summary.json`;
- reloads `output_dir/final` on CPU and records an inference probe.

It does not delete old output. `--allow-existing-output` permits reuse but can leave stale files, so prefer a new run directory.

## Task-aware evaluation

Evaluate the untouched test split after model selection. Compare the base and fine-tuned checkpoints with one command:

```bash
uv run /path/to/gliner2/scripts/evaluate_gliner2.py \
  --data data/test.jsonl \
  --model baseline=fastino/gliner2-base-v1 \
  --model finetuned=./runs/domain-v1/final \
  --output artifacts/test-metrics.json
```

The evaluator reports:

- entity exact mention precision/recall/F1 over `(example, type, mention)`;
- directional binary relation tuple precision/recall/F1;
- single-label accuracy;
- multi-label micro and macro F1;
- structured required-field coverage and normalized exact-instance precision/recall/F1;
- unsupported custom-field relation instances;
- API/runtime errors separately from semantic quality;
- a macro summary delta from the first to last model.

Missing predictions count as misses. They are never replaced with zero-like field values. String comparison normalizes case and whitespace, so add domain-specific normalization separately when dates, identifiers, units, or money require it.

For model-free metric verification, provide aligned predictions:

```jsonl
{"prediction":{"entities":{"person":["Ada"]},"classifications":{},"relations":{},"structures":{}}}
```

```bash
uv run /path/to/gliner2/scripts/evaluate_gliner2.py \
  --data data/test.jsonl \
  --predictions predictions.jsonl
```

Custom relation fields are valid training data, but the current public relation inference schema is binary head/tail. The evaluator exposes those instances as unsupported instead of silently converting or scoring them.

## Operational checklist

- Validate the complete corpus and all final splits.
- Verify that validation and test have no source-group leakage.
- Record a base-model baseline before any training.
- Confirm actual trainer and parameter devices.
- Resolve precision explicitly; never inherit FP16 on CPU.
- Ensure evaluation data exists before enabling early stopping or best-model selection.
- Run one optimizer-step smoke and reload `final`.
- Compare task-aware metrics and error slices on bounded training.
- Use a fresh output directory for every experiment.
- Treat checkpoint loading as weights-only warm start.
- Keep Hub publication as a separate, explicit operation after evaluation.
