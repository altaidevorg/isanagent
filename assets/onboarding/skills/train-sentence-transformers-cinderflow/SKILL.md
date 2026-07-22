---
name: train-sentence-transformers-cinderflow
description: Train or fine-tune sentence-transformers models with Cinderflow instead of HF Jobs. Use for SentenceTransformer, CrossEncoder, or SparseEncoder training tasks that should run from IsanAgent through the public Cinderflow client on configured GPUs, especially ThunderCompute. Covers model-type routing, loss/evaluator selection, production script templates, workflow-submit training with durable logs/artifacts, exec-only preflight diagnostics, Hugging Face Hub publishing, and paid GPU cleanup.
---

# Train a sentence-transformers Model with Cinderflow

**This SKILL.md is a router, not a manual.** It tells you which references and example scripts to load for the task. The detailed sentence-transformers guidance lives in `references/` and `scripts/`; Cinderflow execution guidance lives in `references/cinderflow_execution.md`.

**Do not synthesize a training script from this file alone.** Open the per-type production template (`scripts/train_<type>_example.py`) and copy it as your starting point. The templates contain load-bearing scaffolding (autocast helper, model-card class, logger silencing list, `force=True`, `seed`, TF32, version-compatible imports, named-evaluator metric handling) that prior agent runs have repeatedly missed when rolling their own from a synthesized snippet.

**Also use the Cinderflow client operator skill before provisioning or running.** This skill decides how to build the sentence-transformers training script; the Cinderflow skill governs GPU lifecycle, `exec`, file transfer, logs, and paid GPU cleanup.

## 1. Identify the model type

| Tag | Class | What it does | When to pick |
|---|---|---|---|
| **[SentenceTransformer]** | `SentenceTransformer` (bi-encoder) | Maps each input to a fixed-dim dense vector | Retrieval, similarity, clustering, classification, paraphrase mining, dedup |
| **[CrossEncoder]** | `CrossEncoder` (reranker) | Scores `(query, passage)` pairs jointly | Two-stage retrieval (rerank top-100 from bi-encoder), pair classification |
| **[SparseEncoder]** | `SparseEncoder` (SPLADE) | Sparse vectors over the vocabulary | Learned-sparse retrieval, inverted-index backends (Elasticsearch / OpenSearch / Lucene) |

Tiebreakers when the request is ambiguous: "embedding model" / "vector search" / "similarity" -> **[SentenceTransformer]**. "rerank" / "ranker" / "two-stage" -> **[CrossEncoder]**. "SPLADE" / "sparse" / "inverted index" -> **[SparseEncoder]**. If still unclear, ask.

## 2. Required reading

**Read these in full before writing any code. Do not triage by perceived relevance.**

### Per-type — always required

**[SentenceTransformer]**
- `references/losses_sentence_transformer.md` — loss-to-data-shape mapping; `BatchSamplers.NO_DUPLICATES` requirement for MNRL-family; `Cached*` <-> `gradient_checkpointing` incompatibility.
- `references/evaluators_sentence_transformer.md` — evaluator-to-task mapping; `metric_for_best_model` key construction (named vs unnamed); per-evaluator `primary_metric` values.
- `references/model_architectures.md` — encoder vs decoder vs static vs Router pipelines; pooling rules (mean / cls / lasttoken); auto-mean-pooling behavior for fresh-start MLM bases.
- `scripts/train_sentence_transformer_example.py` — production template; copy this as your starting point.

**[CrossEncoder]**
- `references/losses_cross_encoder.md` — pointwise / pairwise / listwise / distillation; `pos_weight` derivation; `activation_fn=Identity()` mandatory for non-BCE losses (silent eval-rank collapse otherwise).
- `references/evaluators_cross_encoder.md` — `CrossEncoderRerankingEvaluator` recipe; named-evaluator key format `eval_{name}_{primary_metric}`.
- `scripts/train_cross_encoder_example.py` — production template; copy this as your starting point.

**[SparseEncoder]**
- `references/losses_sparse_encoder.md` — `SpladeLoss` wrapper requirement; FLOPS regularizer weights; smoke-test active-dim ramp behavior.
- `references/evaluators_sparse_encoder.md` — `SparseNanoBEIREvaluator` (English-only) and the in-domain alternative; `eval_{name}_{primary_metric}` key format.
- `scripts/train_sparse_encoder_example.py` — production template; copy this as your starting point.

### Cross-cutting — always required

- `references/training_args.md` — `TrainingArguments` knobs, precision rules, warmup/save/eval constraints, schedulers, HPO, tracker, resume, hub-push variants.
- `references/dataset_formats.md` — column-matching rules, reshaping recipes, hard-negative mining options.
- `references/base_model_selection.md` — model discovery, per-type namespaces, ModernBERT-family `max_seq_length=8192` trap, dataset loader pitfalls.
- `references/troubleshooting.md` — symptom-indexed failure recipes; skim headings on every run.
- `references/cinderflow_execution.md` — required for every Cinderflow/ThunderCompute run.

### Cross-cutting — load when applicable

- `references/hardware_guide.md` — VRAM sizing, multi-GPU, FSDP / DeepSpeed. Read for >24GB models or multi-GPU planning, but do not use its HF Jobs execution path for this skill.
- `references/prompts_and_instructions.md` — required when using prompt-tuned bases (E5, BGE, GTE, Qwen3-Embedding, Instructor, Nomic, etc.) or adding `query: ` / `passage: ` prefixes.

Do not use `references/hf_jobs_execution.md` for this Cinderflow variant unless the user explicitly asks to compare against HF Jobs. Cinderflow + ThunderCompute is the default compute path.

### Variant scripts (open when the task matches)

- **[SentenceTransformer]** `scripts/train_sentence_transformer_<matryoshka|multi_dataset|with_lora|distillation|make_multilingual|static_embedding>_example.py`.
- **[CrossEncoder]** `scripts/train_cross_encoder_<distillation|listwise>_example.py`.
- **[SparseEncoder]** `scripts/train_sparse_encoder_distillation_example.py`.
- Hard-negative mining CLI — `scripts/mine_hard_negatives.py`.

## 3. Defaults

Override only if the user specifies otherwise:

- **Cinderflow workflow execution on ThunderCompute via `workflow submit`.** Use the public `cinderflow` CLI against an already-running Cinderflow API server. For this skill, the default training path is a generated repo-less native Cinderflow workflow (`execution: native` without `uses`) that creates a local `uv` virtual environment on the remote GPU. Do not default to a long `cinderflow exec` command.
- **Smoke test first.** Use `max_steps=1` and a tiny dataset slice before any long run.
- **Single run.** After it completes, propose experimentation if the verdict is weak/marginal.
- **Public Hub push at end-of-run, wrapped in try/except.** HF Hub publishing remains supported; HF Jobs execution is not used.
- **Workflow-first smoke outputs.** For smoke tests and short iterations, submit a workflow with small declared artifacts first: `train_log` and `verdict`. Do not declare the full model directory as a workflow artifact in the first smoke run.
- **Cleanup paid GPUs.** Remove ThunderCompute GPUs after the run unless the user explicitly says to keep them.


## Cinderflow-specific hard stops

These rules prevent the most common failed agent-generated workflows:

- Do not write a scratch sentence-transformers training script from memory. Copy
  the selected production template first. If the required `scripts/` templates are
  not available in the installed skill package, stop and ask for those assets
  instead of inventing a training script.
- Do not use `pip install sentence-transformers` or
  `pip install sentence-transformers[train]>=5.0` by itself on a remote GPU. That
  can install a `transformers` version that disables the existing PyTorch
  install.
- The default workflow step for sentence-transformers smoke training must be `execution: native` without `uses`. The generated `train.py` should include PEP 723 inline script metadata and run with `uv run --script train.py`; use explicit `.venv` creation only for dependency-resolution debugging. The script must install or verify a compatible stack before importing sentence-transformers:
  - `torch>=2.6,<2.8` from the CUDA 12.4 wheel index (`https://download.pytorch.org/whl/cu124`) unless the user explicitly chooses a newer tested stack. Do not exact-pin `torch==2.6.0` unless debugging reproducibility.
  - `transformers<5`.
  - `sentence-transformers[train]`, `datasets`, `accelerate`,
    `scikit-learn`. Do not install `trackio` for smoke tests unless the user explicitly asks for HF/Trackio experiment tracking and provides HF auth.
- After dependency installation, run a Python preflight that imports `torch`,
  `transformers`, and `sentence_transformers`, prints versions, verifies CUDA
  availability when a GPU run is expected, and fails immediately if PyTorch is
  unavailable or below 2.6.
- If logs contain `Disabling PyTorch`, `PyTorch was not found`, or
  `Models won't be available`, stop the run and fix dependencies. Do not treat
  this as a Cinderflow or ThunderCompute instability.
- If logs contain `Due to a serious vulnerability issue in torch.load` or `upgrade torch to at least v2.6`, the dependency stack is too old. Regenerate the workflow with `torch>=2.6,<2.8` from `https://download.pytorch.org/whl/cu124`; do not debug Cinderflow runtime or retry the same workflow unchanged.
- Prefer `cinderflow workflow submit` for smoke tests and real training runs. For generated one-file smoke scripts, prefer repo-less native workflow execution with `uv`, not container execution and not `exec`.
- After provisioning ThunderCompute, set `GPU_ID` exactly to the returned `thundercompute-*` GPU id. Before submit, print `GPU_ID`, run `cinderflow gpu status "$GPU_ID" --json`, and ensure the workflow submit command uses `--gpu "$GPU_ID"`. Never submit sentence-transformers smoke tests to stale SSH GPUs such as `ssh-54.161.35.164-0`.
- Use `cinderflow exec` only for short preflight diagnostics, workspace checks, CUDA checks, or emergency debugging. Do not run dependency installation plus training as one long `exec` command.
- Do not treat a non-updating operation as failed during the initial remote setup window. Fresh GPUs may spend several minutes installing Cinderflow utilities, downloading images/packages, creating environments, pulling Docker images, and starting the remote runner before logs advance.
- Minimum wait rule: after `workflow submit`, run `cinderflow operations wait "$OPERATION_ID" --timeout 1800 --poll-interval 30 --json`. Do not replace it with repeated `operations get`. If wait returns `data.job_id`, switch to `workflow logs --tail 400 --follow`.
- Do not remove the GPU just because `operations wait` has not returned yet. If the operation/job is still non-terminal after the wait timeout and the user did not pre-authorize aborting, stop and ask the user before `cinderflow gpu rm`. Do not invent labels like doom loop or infrastructure failure before a terminal error or user-approved timeout cleanup.
- A workflow-based run is not complete unless it reaches `SUCCEEDED`, logs show the training command actually ran, and the `train_log` plus `verdict` artifacts can be listed/downloaded and verified.
- Never mark training validated from `SUCCEEDED` alone. If logs show `Running container with default image command`, `command_preview: default`, `outputs: []`, empty step logs, or missing `VERDICT:`, the workflow did not run the training script even if the job state is `SUCCEEDED`.
- For pure infrastructure testing, use the Cinderflow skill and
  `examples/dummy_workflow.yaml`; do not create a fake sentence-transformers
  workflow that bypasses the production-template contract.

## 4. Constraints the produced script must satisfy

These are non-negotiable contracts. Implementation lives in the production templates and references — do not reinvent.

- Capture the pre-training evaluator score as `baseline_eval` **before** `trainer.train()`.
- Emit a single end-of-run line: `VERDICT: WIN|MARGINAL|REGRESSION | score=... | baseline=... | delta=...`. Cinderflow logs are monitored for this.
- Also write the verdict and key metrics to a file under the workflow output directory, e.g. `${CINDERFLOW_OUTPUTS}/metrics/verdict.txt`.
- Silence `httpx`, `httpcore`, `huggingface_hub`, `urllib3`, `filelock`, `fsspec` to WARNING.
- Tee logs to `${CINDERFLOW_OUTPUTS}/logs/train.log` so `cinderflow workflow logs` and the `train_log` artifact expose the same training output.
- End with `model.push_to_hub(...)` wrapped in `try/except`.
- Smoke-test before any long run (`max_steps=1` + tiny dataset slice).
- **[CrossEncoder]** Include `EarlyStoppingCallback(patience>=3)`.
- **[SparseEncoder]** Log `query_active_dims` / `corpus_active_dims` on the verdict line; use suffix matching for name-prefixed metric keys.


## Exec vs workflow decision rule

Default to this lifecycle:

1. Use `cinderflow exec` only for short preflight diagnostics: remote shell works,
   workspace exists, `uv` exists, and `nvidia-smi` sees the GPU.
2. Generate a Cinderflow workflow for the smoke training run.
3. Validate the workflow before submitting it, then inspect the YAML for the required dependency/output rules. Validation alone is not enough.
4. Confirm `GPU_ID` is the newly provisioned `thundercompute-*` id, not a stale SSH GPU id.
5. Submit with `cinderflow workflow submit`, then run `cinderflow operations wait "$OPERATION_ID" --timeout 1800 --poll-interval 30 --json`. Do not treat a quiet `RUNNING` operation as failure while the wait command is still within its timeout.
6. Once a `job_id` exists, follow logs with `cinderflow workflow logs "$JOB_ID" --tail 400 --follow`.
7. Download or inspect the declared `train_log` and `verdict` artifacts after the
   workflow reaches a terminal state.

Rationale:

- Long `cinderflow exec` commands are fragile: client HTTP timeouts, quoting
  errors, and chained dependency installs make failures hard to diagnose.
- `workflow submit` is the stable training path because it adds autonomous remote
  execution, job state, runner logs, container logs, operation history, SSE log
  following, and declared artifacts.
- `exec` remains useful for tiny diagnostics and emergency debugging, but it is
  not the default training runner.

Workflow artifact rules:

- Start workflow artifacts with small outputs only: `train_log` and `verdict`.
- Do not declare the full model directory as a workflow artifact in the first
  smoke workflow. Large model-directory upload can fail or be slow; prefer HF Hub,
  S3, or a later dedicated artifact strategy for full model outputs.
- If model upload is required, make it explicit and handle permissions/size as a
  separate step after the smoke workflow succeeds.



## Robust script embedding and Python environment isolation

Never install sentence-transformers dependencies into the remote system Python.
Use a script-owned dependency contract and keep execution isolated from system and
user site packages.

For generated Python training scripts, use PEP 723 inline script metadata at the
top of `train.py`. This makes the script carry its dependency contract and keeps
the workflow shell shorter and less error-prone.

Default metadata pattern:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.6,<2.8",
#   "transformers<5",
#   "sentence-transformers[train]",
#   "datasets>=2.18.0,<4",
#   "huggingface-hub>=0.21.2,<1",
#   "fsspec>=2023.12.0,<2025",
#   "pyarrow<19",
#   "accelerate",
#   "scikit-learn",
# ]
# ///
```

Rules for metadata:

- Include this metadata in every generated sentence-transformers smoke/training
  script.
- Keep `torch>=2.6,<2.8`; do not exact-pin `torch==2.6.0` unless reproducing a
  specific failure.
- Keep `transformers<5` until the templates and callbacks are verified against
  Transformers v5.
- Pin `pyarrow<19` to resolve PyArrow compatibility issues (such as PyExtensionType errors).
- Use bounded dataset stack dependencies: `datasets>=2.18.0,<4`, `huggingface-hub>=0.21.2,<1`, and `fsspec>=2023.12.0,<2025` to avoid known fsspec double-asterisk path pattern errors while preventing untested future major versions.
- Do not include `trackio` unless the user explicitly asks for HF/Trackio
  experiment tracking and provides the needed HF auth.
- Still run explicit import/CUDA preflight in the script; metadata does not
  prove the resolved stack is usable.

Do not place long Python training scripts directly into YAML block scalars.
Base64-encode the adapted training script locally and decode it inside the
workflow step. This is mandatory for generated multi-line Python scripts in this
skill. This avoids YAML indentation, quoting, backslash, and brace fragility.

Before submitting any Base64-embedded workflow, validate the payload locally:

```bash
python -m py_compile train.py
base64 -i train.py | tr -d '\n' > train.py.b64
base64 -d train.py.b64 > /tmp/decoded_train.py
python -m py_compile /tmp/decoded_train.py
```

Rules for Base64 payloads:

- Paste only the single-line contents of `train.py.b64` into the workflow YAML.
- Do not leave placeholders such as `<BASE64_ENCODED_SCRIPT>` in submitted YAML.
- Do not paste markdown fences, quotes, comments, or explanatory text into the payload.
- Do not let the payload wrap into multiple YAML lines unless the decode command explicitly reconstructs it safely.
- Use `printf '%s' '<BASE64_STRING>' | base64 -d > train.py` in the workflow step.
- Do not use `echo '<BASE64_STRING>' | base64 -d > train.py`; `echo` can add or interpret characters in shell-dependent ways.
- If logs show `base64: invalid input`, the training did not start. Fix the workflow generation/payload first; do not debug PyTorch, uv, CUDA, or Cinderflow runtime.

Default native workflow pattern:

```bash
set -euo pipefail
out="${CINDERFLOW_OUTPUTS:-/outputs}"
work="$HOME/.cinderflow/workspaces/default/sentence-transformers/<run_name>-${CINDERFLOW_JOB_ID:-manual}"
mkdir -p "$work" "$out/model" "$out/logs" "$out/metrics"
cd "$work"
printf '%s' '<BASE64_ENCODED_SCRIPT>' | base64 -d > train.py
python -m py_compile train.py
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
PYTHONNOUSERSITE=1 \
UV_INDEX_URL="https://pypi.org/simple" \
UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124" \
uv run --script train.py 2>&1 | tee "$out/logs/train.log"
```

Why this is preferred:

- `uv run --script train.py` reads the PEP 723 metadata and creates an isolated
  script environment automatically.
- The workflow no longer needs a long sequence of `uv venv` and `uv pip install`
  commands that agents often mutate incorrectly.
- `UV_EXTRA_INDEX_URL` exposes the PyTorch CUDA wheel index while keeping PyPI
  available for normal packages.
- `PYTHONNOUSERSITE=1` prevents user/system site packages from leaking into the
  script environment.

Use the older explicit `.venv` + `uv pip install --python .venv/bin/python ...`
pattern only for debugging dependency resolution or when `uv run --script` is not
available on the remote host.

Use a container workflow only when the user explicitly needs a specific image or
Docker-level reproducibility. For container workflow steps, prefer `python3 -m
venv` because standard PyTorch/framework images do not always include `uv`.

Rules:

- Do not run bare `pip install ...` or bare `python3 train.py` for training.
- For native workflow smoke tests, prefer PEP 723 metadata plus `uv run --script train.py`.
- Expected known-good smoke environment from ThunderCompute A6000: CUDA-visible PyTorch such as `torch>=2.6` with CUDA support, `transformers= 4.57.x` or another compatible `<5` version, `sentence_transformers= 5.6.x` or another compatible version, and `cuda_available= True`.
- Do not install `torch==2.4.0` or any `torch<2.6` for current transformer-based sentence-transformers runs. Prefer `torch>=2.6,<2.8` from the CUDA 12.4 wheel index and let `uv` resolve the exact compatible wheel, or keep a newer known-good existing isolated script environment.
- Do not uninstall system packages such as `torchvision` from `/usr`,
  `/opt/conda`, or system `dist-packages`. If system `torchvision` causes
  `RuntimeError: operator torchvision::nms does not exist`, the fix is to stop
  using system Python and recreate the isolated script environment.
- Set `PYTHONNOUSERSITE=1` for preflight and training commands so user/system
  site packages do not leak into the smoke environment.
- Do not clear the remote `uv` cache by default. Use `uv cache clean` or remove uv lock files only when logs explicitly show uv cache lock, filelock, corrupted-cache, or in-use cache errors. Prefer run-local cache (`UV_CACHE_DIR="$work/.uv-cache"`) when iterating dependency stacks.
- During iterative debugging, change `metadata.name` and pass a unique `run_id` input (timestamp or UUID) so Cinderflow cannot accidentally reuse previous step results. Do not describe the exact cache key as stable API behavior.

## Required exec diagnostics before workflow submit

Before submitting the workflow, run and record these small `cinderflow exec` checks. Do not skip them on a fresh ThunderCompute GPU:

```bash
cinderflow exec --gpu "$GPU_ID" 'whoami && pwd && echo "$HOME"'
cinderflow exec --gpu "$GPU_ID" 'mkdir -p "$HOME/.cinderflow/workspaces/default/sentence-transformers/<run_name>" && test -d "$HOME/.cinderflow/workspaces/default/sentence-transformers/<run_name>" && echo workspace-ok'
cinderflow exec --gpu "$GPU_ID" 'python3 --version || python --version'
cinderflow exec --gpu "$GPU_ID" 'uv --version || true'
cinderflow exec --gpu "$GPU_ID" 'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader'
```

If any of these fail, do not diagnose it as a sentence-transformers dependency
problem. Treat it as Cinderflow exec/SSH/workspace/runtime readiness and report
the exact failing command.

Exit code `255` is not enough to conclude Python or pip is broken. It often means
SSH/channel/remote command failure, command quoting failure, or a timeout around a
long command. After a `255` or `timed out` result:

- Do not open a new GPU immediately.
- Run `cinderflow gpu ls --json` and check whether the current GPU is still
active.
- Retry only the smallest diagnostic command, e.g. `cinderflow exec --gpu
  "$GPU_ID" 'echo cinderflow-exec-ok'`.
- Do not run dependency installation plus training through `exec`. Fix the workflow or use `exec` only for the smallest failing diagnostic command.



## Dataset and memory guardrails

Default smoke runs must be bounded. Do not materialize large or unknown datasets
in full during smoke tests.

Hard rules:

- For smoke tests, never call `load_dataset(..., split="train")` on large datasets
  and then process the entire split.
- For known large datasets such as MS MARCO, HotpotQA, Natural Questions, or any
  dataset likely to exceed 100K rows, use one of these bounded paths for smoke/default runs:
  - `streaming=True` plus `itertools.islice(...)` / `.take(N)`.
  - A small explicit split slice such as `train[:100]` when the dataset supports
    efficient slicing.
  - A pre-filtered local/S3/HF subset prepared before the training run.
- For smoke/default runs, loading massive datasets (like the 8.8M row MS MARCO corpus) entirely into memory can cause OOM failures and hangs. Optimize dataset resolution logic by filtering the corpus and queries by required unique IDs in batches before mapping, rather than loading the entire corpus. Full materialization is allowed only for an explicit long run with a memory/runtime plan.
- Explicitly cast Hugging Face dataset columns to standard Python lists (e.g., `list(dataset['column'])`) to prevent TypeErrors in newer Hugging Face dataset versions.
- Smoke workflows must expose explicit limits such as `train_size`, `eval_size`,
  `query_limit`, `passage_limit`, or `max_steps` and must use those limits in the
  script.
- Do not resolve all query/passage/document IDs from a large dataset for a smoke
  test. Bound the unique ID set first.
- A full-dataset run is allowed only after the user explicitly asks for a long
  run and the agent has a memory/storage/runtime plan.

If logs show a huge split generation such as millions of examples, do not assume
it is stuck immediately, but do treat it as a warning that the script may be
materializing too much data. Prefer regenerating the script with streaming and
bounded limits for smoke tests.

## Hugging Face Dataset column datatype rules

Hugging Face `datasets` columns may be `Column` or sequence-like objects, not
plain Python lists. Do not concatenate or mutate dataset columns directly.

Wrong:

```python
passage_ids = set(raw["positive_id"] + raw["negative_id"])
queries = raw["query"]
```

Correct:

```python
positive_ids = list(raw["positive_id"])
negative_ids = list(raw["negative_id"])
passage_ids = set(positive_ids + negative_ids)
queries = list(raw["query"])
```

For optional or nested IDs, use a safe flattener:

```python
def flatten_ids(values):
    out = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            out.extend(x for x in value if x is not None)
        else:
            out.append(value)
    return out
```

If logs show `TypeError: unsupported operand type(s) for +: 'Column' and
'Column'`, this is a training-script dataset handling bug. Fix the script by
casting columns to lists before retrying. Do not debug Cinderflow runtime or open
a new GPU for the same broken script.

## MS MARCO and CrossEncoder distillation guardrails

MS MARCO-style CrossEncoder distillation is especially easy to make too large or
memory-heavy. For smoke runs:

- Use streaming or a tiny slice for triplets/pairs.
- Bound `TRAIN_SIZE`, `EVAL_SIZE`, and any ID resolution limits.
- Resolve query IDs and passage IDs only for the bounded sample.
- Cache resolved small samples in the run workspace if the script retries within
  the same job.
- Do not build a full corpus map for millions of passages in a smoke workflow.
- Do not assume `positive_id`, `negative_id`, `query_id`, or passage columns are
  Python lists; cast them explicitly.
- If a model needs non-safetensors `.bin` weights, ensure the dependency stack
  uses PyTorch >=2.6 as described above.

A CrossEncoder smoke success requires both infrastructure success and script
success: CUDA visible, bounded dataset processed, training step executed,
`VERDICT:` emitted, and `train_log`/`verdict` artifacts registered.

## Generated workflow self-checks

Before `cinderflow workflow validate` and `workflow submit`, inspect generated
workflow YAML and stop if any of these checks fail:

```bash
if rg -q '\$\{inputs\.' workflow.yaml; then exit 1; fi
if rg -q '\$\{ inputs\.' workflow.yaml; then exit 1; fi
if rg -q '<BASE64' workflow.yaml; then exit 1; fi
if rg -q 'echo "[A-Za-z0-9+/=]{80,}" \| base64 -d' workflow.yaml; then exit 1; fi
```

Interpretation:

- `${inputs...}` or `${ inputs...}` means invalid Altai substitution syntax.
- `<BASE64...>` means a placeholder was submitted instead of a real payload.
- `echo "..." | base64 -d` means the generated workflow used the fragile decode
  pattern. Use `printf '%s' '...' | base64 -d` instead.

These self-checks are not a substitute for Cinderflow validation. Run both the
self-checks and `cinderflow workflow validate <workflow.yaml> --json`.

## Altai input substitution guardrails

Cinderflow Altai v1 resolves workflow inputs only with double-brace syntax:

```yaml
with:
  train_size: ${{ inputs.train_size }}
  max_steps: ${{ inputs.max_steps }}
  model_name: ${{ inputs.model_name }}
```

Inside the shell command, read the resolved values from `ALTAI_PARAM_*`:

```bash
TRAIN_SIZE="${ALTAI_PARAM_TRAIN_SIZE}" MAX_STEPS="${ALTAI_PARAM_MAX_STEPS}" MODEL_NAME="${ALTAI_PARAM_MODEL_NAME}" PYTHONNOUSERSITE=1 uv run --script train.py
```

Never use these invalid forms:

```yaml
train_size: ${inputs.train_size}
train_size: ${ inputs.train_size }
```

Never put invalid forms directly in shell env assignment:

```bash
TRAIN_SIZE="${inputs.train_size}"
TRAIN_SIZE="${ inputs.train_size }"
```

Before submit, inspect the generated YAML and stop if any invalid substitution
pattern is present:

```bash
if rg -q '\$\{inputs\.' workflow.yaml; then exit 1; fi
if rg -q '\$\{ inputs\.' workflow.yaml; then exit 1; fi
```

If logs show `ValueError: invalid literal for int() with base 10: '${inputs...}'`
or similar, the workflow input was not substituted. Regenerate the workflow YAML
with `${{ inputs.name }}` in the `with` map and `ALTAI_PARAM_*` in the shell. Do
not retry the same workflow unchanged.

## 5. Cinderflow workflow run

Default to this flow for sentence-transformers smoke tests and training runs:

1. Identify the model type (§1).
2. Load the required references and production template (§2).
3. Copy the matching `scripts/train_<type>_example.py` as the starting point.
4. Adapt `MODEL_NAME`, `DATASET_NAME`, `RUN_NAME`, loss, evaluator, and trainer args.
5. Make the script write logs and verdict files under `CINDERFLOW_OUTPUTS`, for example `${CINDERFLOW_OUTPUTS}/logs/train.log` and `${CINDERFLOW_OUTPUTS}/metrics/verdict.txt`.
6. Provision or select exactly one GPU following the Cinderflow skill rules.
7. Run only the required short exec diagnostics before submit; do not run dependency installation plus training through `exec`.
8. Generate an Altai v1 Cinderflow workflow YAML that runs the adapted script in a repo-less native step by default: `execution: native` without `uses`. Base64-decode the script and create an isolated `uv` environment inside the step. Use a container step only when the user explicitly asks for containerized execution or a specific image.
9. Inspect the generated YAML before submit. It must not contain bare `pip install sentence-transformers`, bare `python train.py`, `trackio`, `uses:` for the default smoke path, or a full model-directory artifact in the first smoke run. It must define `train_log` and `verdict` outputs, and the training command must tee stdout/stderr to `${CINDERFLOW_OUTPUTS}/logs/train.log`.
   It must use Altai v1 double-brace input substitution, e.g. `${{ inputs.max_steps }}`. Never use `${ inputs.max_steps }`, which remains a literal string and can cause `ValueError: invalid literal for int() with base 10`.
   - The generated workflow must decode the Base64 payload with `printf '%s'`, not `echo`. A failure `base64: invalid input` means the workflow generation is wrong and the training script never started.
   - Before validation/submission, reject generated YAML that contains `${inputs.` or `${ inputs.`. Altai v1 input substitution must use double braces such as `${{ inputs.train_size }}`. Single-brace forms are passed through literally and can crash Python with errors like `ValueError: invalid literal for int() with base 10: '${inputs.train_size}'`.
   - Prefer passing input values through the step `with` map and reading the corresponding `ALTAI_PARAM_*` environment variables inside the shell command. Example: set `train_size: ${{ inputs.train_size }}` in `with`, then run `TRAIN_SIZE="${ALTAI_PARAM_TRAIN_SIZE}" uv run --script train.py`.

10. Validate the workflow with `cinderflow workflow validate <workflow.yaml> --json`.
11. Confirm `GPU_ID` is the exact returned `thundercompute-*` id with `cinderflow gpu status "$GPU_ID" --json`; do not use stale SSH GPU ids.
12. Submit with `cinderflow workflow submit <workflow.yaml> --gpu "$GPU_ID" --json`.
13. Wait for the returned `operation_id` with `cinderflow operations wait "$OPERATION_ID" --timeout 1800 --poll-interval 30 --json`. Use `data.job_id` from the wait result.
14. Wait for `operations wait` to finish or time out before declaring a non-terminal `RUNNING` operation stuck. Do not run tight `operations get` loops.
15. Once `JOB_ID` exists, follow logs with `cinderflow workflow logs "$JOB_ID" --tail 400 --follow`. If logs are quiet, continue polling `workflow status` and `workflow logs --tail 400`; remote setup can still be running.
16. Do not abandon workflow mode or delete the GPU during the initial setup window. If it is still `RUNNING` after 20 minutes and there is no terminal error, ask the user before cleanup unless the user pre-authorized a timeout budget.
17. Verify terminal state with `cinderflow workflow status "$JOB_ID" --json`, then verify real training success: logs must contain `VERDICT:`, logs must show `Running step ... via UV-native (no Docker)` for the default native path, logs must not show default container command/no outputs, `artifacts ls` must include the step's `train_log` and `verdict` refs such as `train_native.train_log` and `train_native.verdict`, and the downloaded verdict file must contain `VERDICT: WIN|MARGINAL|REGRESSION`.
18. Remove paid GPUs with `cinderflow gpu rm "$GPU_ID" --json` only after terminal success/failure, explicit user abort, or user-approved timeout cleanup; then verify no active ThunderCompute instance remains.
19. After the run, append to `logs/experiments.md` locally if the working tree/task has such a log file, and propose iteration if the verdict is weak/marginal.

Use `cinderflow exec` for debugging only. If workflow logs reveal an environment issue, reproduce the smallest failing command with `exec`; do not replace the workflow run with one long `exec` command. Do not switch to exec just because operation polling is quiet during remote setup. Do not provision a replacement GPU for manual exec diagnostics until the current workflow reaches a terminal state or the user approves aborting it.

## Required post-success verification

A `SUCCEEDED` job is not enough to claim sentence-transformers training succeeded. Before reporting success, run and verify:

```bash
cinderflow workflow status "$JOB_ID" --json
cinderflow workflow logs "$JOB_ID" --tail 400
cinderflow artifacts ls "$JOB_ID" --json
cinderflow artifacts download --job "$JOB_ID" --artifact train_native.verdict --output ./verdict.txt
cinderflow artifacts download --job "$JOB_ID" --artifact train_native.train_log --output ./train.log
```

Success requires all of these:

- Job state is `SUCCEEDED`.
- Logs contain `VERDICT:` from the training script.
- Logs do not contain `Running container with default image command`, `command_preview: default`, or `outputs: []`.
- `artifacts ls` includes both verdict and train log refs, for example `train_native.verdict` and `train_native.train_log`.
- `./verdict.txt` contains `VERDICT: WIN`, `VERDICT: MARGINAL`, or `VERDICT: REGRESSION`.

If any check fails, report the run as infrastructure/workflow generation failure, not as a successful training smoke test.

## Prerequisites

Client side:

```bash
CINDERFLOW_API_BASE_URL=http://localhost:8070
CINDERFLOW_API_TOKEN=<jwt>
cinderflow --help
```

Server/remote side:

- Cinderflow API server is already running.
- Provider credentials live on the server.
- `HF_TOKEN` with write scope is available to the runner if Hub push is requested.
- The workflow step installs or verifies the pinned stack from `references/cinderflow_execution.md`: compatible PyTorch, `transformers<5`, `sentence-transformers[train]`, datasets/training dependencies without `trackio` by default, and the import/CUDA preflight.

Do not ask the public client user for provider secrets, SSH keys, Jupyter tokens, database URLs, or server internals unless the user explicitly switches to server-admin work.
