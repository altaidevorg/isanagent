# Cinderflow Execution for sentence-transformers Training

Use this reference whenever the sentence-transformers training run should execute through Cinderflow instead of HF Jobs.

## Execution model

For sentence-transformers smoke tests and training runs, use `cinderflow workflow submit` by default. Long `cinderflow exec` commands are too fragile for dependency installation plus training because they are easy to misquote and can hit client-side request timeouts.

## Robust workflow script pattern

When a workflow step needs a generated Python training script, Base64-encode the
script locally and decode it in the step command:

```bash
printf '%s' "<BASE64_ENCODED_SCRIPT>" | base64 -d > train.py
python -m py_compile train.py
```

Use `printf '%s'`, not `echo`, because `echo` can add or interpret characters in
shell-dependent ways and cause `base64: invalid input`. This is preferred over
embedding Python source directly in YAML block scalars, which is fragile under
indentation, quoting, backslash, and brace changes.

For container workflow steps, use `python3 -m venv .venv` unless the selected
image explicitly includes `uv`. Standard PyTorch/framework images usually do not
ship with `uv`. Prefix dependency installation, preflight, and training commands
with `PYTHONNOUSERSITE=1` to prevent system/user site-packages from leaking into
the isolated environment.

Always tee training output to a declared artifact path. In native mode, use the
same `uv run --script train.py` command for preflight and training:

```bash
PYTHONNOUSERSITE=1 \
UV_INDEX_URL="https://pypi.org/simple" \
UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124" \
uv run --script train.py 2>&1 | tee "${CINDERFLOW_OUTPUTS:-/outputs}/logs/train.log"
```

Declare at least `train_log` and `verdict` outputs for smoke runs.

The agent should produce two local artifacts before running:

1. A training script copied from the correct production template in `scripts/` and adapted to the task.
2. An Altai v1 Cinderflow workflow YAML that runs that script and writes small declared outputs.

Use `cinderflow exec` only for short preflight diagnostics before submit, such as checking `whoami`, the workspace path, `uv`, and `nvidia-smi`. Do not run dependency installation plus training through one long `exec` command.

Recommended workflow output layout:

```text
${CINDERFLOW_OUTPUTS}/
  logs/train.log
  metrics/verdict.txt
```

In the first workflow smoke, declare only log/verdict outputs; do not declare the full model directory as an artifact.


## Workflow-first policy

For sentence-transformers work, the default path is:

1. Run short `cinderflow exec` diagnostics only.
2. Generate and validate the workflow YAML, then inspect it for the required dependency/output rules. Validation alone is not enough.
3. Confirm `GPU_ID` is the newly provisioned `thundercompute-*` id with `cinderflow gpu status "$GPU_ID" --json`.
4. Submit the workflow with `cinderflow workflow submit --gpu "$GPU_ID"`; never use stale SSH GPU ids such as `ssh-54.161.35.164-0`.
5. Run `cinderflow operations wait "$OPERATION_ID" --timeout 1800 --poll-interval 30 --json`. Repeated `RUNNING` records or missing `resource_id` during initial setup are not failure while wait is still active.
6. Convert `operation_id` to `job_id`, follow workflow logs, and inspect declared artifacts.

Use `cinderflow exec` after submit only to debug the smallest failing command if workflow logs show an environment or workspace problem.

When using workflow mode, declare small outputs first:

- `train_log` at `/outputs/logs/train.log` with role `logs`.
- `verdict` at `/outputs/metrics/verdict.txt` with role `metrics`.

Do not declare the full model directory as a workflow artifact in the first smoke
workflow. Model directories are large and can hit upload or permission issues.
For full model outputs, prefer Hugging Face Hub/S3 or handle model artifact upload
as a separate, explicit production step after the smoke workflow succeeds.


## Exec diagnostics and timeout handling

Before workflow submit, verify the remote execution environment with
small commands:

```bash
cinderflow exec --gpu "$GPU_ID" 'whoami && pwd && echo "$HOME"'
cinderflow exec --gpu "$GPU_ID" 'mkdir -p "$HOME/.cinderflow/workspaces/default/sentence-transformers/<run_name>" && test -d "$HOME/.cinderflow/workspaces/default/sentence-transformers/<run_name>" && echo workspace-ok'
cinderflow exec --gpu "$GPU_ID" 'python3 --version || python --version'
cinderflow exec --gpu "$GPU_ID" 'uv --version || true'
cinderflow exec --gpu "$GPU_ID" 'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader'
```

For `cinderflow files`, remote paths are relative to the Jupyter workspace root.
For `cinderflow exec`, use the absolute workspace path:

```text
$HOME/.cinderflow/workspaces/default/sentence-transformers/<run_name>
```

If `cinderflow exec` returns exit code `255` or `timed out`, do not guess that
Python, pip, or sentence-transformers is broken. First test whether the remote
command channel still works:

```bash
cinderflow gpu ls --json
cinderflow exec --gpu "$GPU_ID" 'echo cinderflow-exec-ok'
```

Do not respond to this by running dependency installation and training through `exec`. If the diagnostic command channel works, fix or submit the workflow. Use `exec` only for the smallest command needed to reproduce a workflow failure.


## Isolated Python environment requirements

For the default native workflow, use PEP 723 metadata plus `uv run --script train.py`; do not manually invoke `.venv/bin/python`. Use a workspace-local `.venv` only for container mode or explicit dependency-resolution debugging. Do not use system Python or system `pip` for sentence-transformers smoke runs. ThunderCompute images may contain preinstalled `torchvision` builds that are incompatible with the torch version you install; this commonly surfaces as:

```text
RuntimeError: operator torchvision::nms does not exist
```

This is not fixed by uninstalling system `torchvision`, and agents should not try
to modify system packages. For container/debugging fallback, recreate/use an isolated venv and run with
`PYTHONNOUSERSITE=1`:

```bash
cd "$HOME/.cinderflow/workspaces/default/sentence-transformers/<run_name>"
python3 -m venv .venv
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --upgrade pip
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6,<2.8"
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install "transformers<5" "sentence-transformers[train]" "datasets>=2.18.0,<4" "huggingface-hub>=0.21.2,<1" "fsspec>=2023.12.0,<2025" "pyarrow<19" accelerate scikit-learn
PYTHONNOUSERSITE=1 .venv/bin/python - <<'PY'
import torch
import transformers
import sentence_transformers
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"sentence_transformers={sentence_transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]) < (2, 6):
    raise RuntimeError(f"PyTorch >=2.6 required, found {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
PY
```

Never diagnose `torchvision::nms` as a broken GPU or Cinderflow server. It is a
Python environment contamination/version mismatch issue.

## Dependency stack rules

Default PEP 723 metadata:

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

Default run command:

```bash
PYTHONNOUSERSITE=1 \
UV_INDEX_URL="https://pypi.org/simple" \
UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124" \
uv run --script train.py 2>&1 | tee "$out/logs/train.log"
```


Do not rely on whatever PyTorch happens to be preinstalled on the remote host or
container. A common failure mode is installing a new `transformers` release that
requires PyTorch >=2.6 while the image contains an older PyTorch; `transformers`
then disables model execution even though installation succeeds.

Prefer PEP 723 inline script metadata plus `uv run --script train.py`. For CUDA-capable ThunderCompute smoke tests, the script metadata should include the bounded stack below and the workflow should provide `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`. Use explicit `.venv`/pip snippets only for dependency-resolution debugging.

```bash
python3 -m venv .venv
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --upgrade pip
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6,<2.8"
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install "transformers<5" "sentence-transformers[train]" "datasets>=2.18.0,<4" "huggingface-hub>=0.21.2,<1" "fsspec>=2023.12.0,<2025" "pyarrow<19" accelerate scikit-learn
PYTHONNOUSERSITE=1 .venv/bin/python - <<'PY'
import torch
import transformers
import sentence_transformers

version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"sentence_transformers={sentence_transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if version < (2, 6):
    raise RuntimeError(f"PyTorch >=2.6 is required, found {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available for this GPU training run")
PY
```

Do not install `trackio` for smoke tests. It may auto-initialize Hugging Face tracking and fail with `LocalTokenNotFoundError` when no HF token is present. Install it only when the user explicitly asks for tracking and provides HF auth.

If this preflight fails, stop and fix the dependency stack. Do not submit or
continue a long training run.

## Required Cinderflow commands for workflow-based runs

Use the public Cinderflow client CLI only.

```bash
cinderflow gpu discover --provider thundercompute --sort hourly_asc --limit 10 --json
cinderflow gpu configure \
  --provider thundercompute \
  --gpu-type a6000 \
  --mode prototyping \
  --template base \
  --cpu-cores 6 \
  --disk-size-gb 100 \
  --gpu-index 0 \
  --ssh-ready-timeout 300 \
  --poll-interval-sec 5 \
  --json
export GPU_ID=<returned-thundercompute-gpu-id>
test "${GPU_ID#thundercompute-}" != "$GPU_ID"
cinderflow gpu status "$GPU_ID" --json
cinderflow exec --gpu "$GPU_ID" 'whoami && echo "$HOME" && uv --version && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader'
cinderflow workflow validate ./sentence-transformers-smoke.yaml --json
# Inspect the YAML: no bare pip install sentence-transformers, no bare python train.py, no trackio, train_log/verdict outputs present.
cinderflow workflow submit ./sentence-transformers-smoke.yaml --gpu "$GPU_ID" --json
cinderflow operations wait "$OPERATION_ID" --timeout 1800 --poll-interval 30 --json
cinderflow workflow logs "$JOB_ID" --tail 400 --follow
cinderflow workflow status "$JOB_ID" --json
cinderflow artifacts ls "$JOB_ID" --json
cinderflow artifacts download --job "$JOB_ID" --artifact train.verdict --output ./verdict.txt
cinderflow artifacts download --job "$JOB_ID" --artifact train.train_log --output ./train.log
cinderflow gpu rm "$GPU_ID" --json
```

Rules:

- Never use `cinderflow be` for ThunderCompute.
- If `gpu configure` times out, run `cinderflow gpu ls --json` before retrying; a paid GPU may already exist.
- `workflow submit` returns an `operation_id`; use `operations wait` until it returns `data.job_id` and terminal status. On a fresh GPU, this can take several minutes while Cinderflow utilities, images, packages, and runner setup are prepared.
- A quiet or unchanged operation is not automatically failed. Let `operations wait` run until it returns or reaches its timeout before considering a non-terminal `RUNNING` operation stuck.
- Do not delete the GPU while the operation/job is non-terminal unless the user explicitly asks to abort or has pre-approved a timeout cleanup. If `operations wait` times out with no terminal state, report the latest wait/error payload and ask before `gpu rm`.
- If the operation request shows a different `gpu_id` than the newly provisioned `thundercompute-*` id, stop and report the wrong-GPU submission instead of diagnosing ThunderCompute connectivity.
- If `gpu rm` times out, verify with `gpu ls --json` and provider/server-side instance state before retrying.
- Do not clear the remote `uv` cache by default. Use `uv cache clean` or remove uv lock files only when logs explicitly show uv cache lock, filelock, corrupted-cache, or in-use cache errors. Prefer run-local cache (`UV_CACHE_DIR="$work/.uv-cache"`) when iterating dependency stacks.
- During iterative debugging, change `metadata.name` and pass a unique `run_id` input (timestamp or UUID) so Cinderflow cannot accidentally reuse previous step results. Do not describe the exact cache key as stable API behavior.
- For smoke/default runs, loading massive datasets (like the 8.8M row MS MARCO corpus) entirely into memory can cause OOM failures and hangs. Optimize dataset resolution logic by filtering the corpus and queries by required unique IDs in batches before mapping, rather than loading the entire corpus. Full materialization is allowed only for an explicit long run with a memory/runtime plan.
- Explicitly cast Hugging Face dataset columns to standard Python lists (e.g., `list(dataset['column'])`) to prevent TypeErrors in newer Hugging Face dataset versions.

## Pre-validated Altai v1 smoke workflow template

Use this shape as the starting point. Do not generate a workflow from scratch unless the user asks for a materially different structure.

```yaml
apiVersion: altai.dev/v1alpha1
kind: TrainingWorkflow
metadata:
  name: sentence-transformer-smoke
  version: "1.0.0"
  description: Sentence-transformers smoke training with verdict artifact

inputs:
  model_name:
    type: string
    default: sentence-transformers/all-MiniLM-L6-v2
  run_name:
    type: string
    default: st-smoke
  max_steps:
    type: integer
    default: 1

steps:
  - id: train
    execution: container
    image: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
    with:
      command:
        - bash
        - -lc
        - |
          set -euo pipefail
          out="${CINDERFLOW_OUTPUTS:-/outputs}"
          mkdir -p "$out/logs" "$out/metrics"
          printf '%s' '<BASE64_ENCODED_SCRIPT>' | base64 -d > train.py
          python -m py_compile train.py
          python3 -m venv .venv
          PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --upgrade pip
          PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6,<2.8"
          PYTHONNOUSERSITE=1 .venv/bin/python -m pip install "transformers<5" "sentence-transformers[train]" "datasets>=2.18.0,<4" "huggingface-hub>=0.21.2,<1" "fsspec>=2023.12.0,<2025" "pyarrow<19" accelerate scikit-learn
          OUTPUT_DIR="$out" \
          MODEL_NAME="${ALTAI_PARAM_MODEL_NAME}" \
          RUN_NAME="${ALTAI_PARAM_RUN_NAME}" \
          MAX_STEPS="${ALTAI_PARAM_MAX_STEPS}" \
          PYTHONNOUSERSITE=1 .venv/bin/python train.py 2>&1 | tee "$out/logs/train.log"
    outputs:
      train_log:
        type: file
        path: /outputs/logs/train.log
        role: logs
      verdict:
        type: file
        path: /outputs/metrics/verdict.txt
        role: metrics
```

Before submit, inspect the final YAML. The `train` step must have a non-empty command, `train_log` and `verdict` outputs, isolated `python3 -m venv`, `PYTHONNOUSERSITE=1`, and no `trackio`. A workflow that runs the default container command is invalid even if schema validation passes.

Input substitution must use Altai v1 double braces: `${{ inputs.max_steps }}`. Never write `${ inputs.max_steps }`; it can pass through as a literal string and fail at runtime with `ValueError: invalid literal for int() with base 10`. Search the final YAML for `${ inputs.` before submitting.

## Workflow step shape

Default to a repo-less native step for generated sentence-transformers scripts.
Use `uv` through PEP 723 script metadata when running native. Use the container
pattern only when the user explicitly asks for containerized execution or a
specific image; container images usually need `python3 -m venv` because `uv` is
not guaranteed to be installed.

Default native shape:

```yaml
steps:
  - id: train_native
    execution: native
    with:
      command:
        - bash
        - -lc
        - |
          set -euo pipefail
          out="${CINDERFLOW_OUTPUTS:-/outputs}"
          work="$HOME/.cinderflow/workspaces/default/${ALTAI_PARAM_RUN_NAME:-st-native}-${CINDERFLOW_JOB_ID:-manual}"
          mkdir -p "$work" "$out/logs" "$out/metrics"
          cd "$work"
          printf '%s' '<BASE64_ENCODED_SCRIPT>' | base64 -d > train.py
          python -m py_compile train.py
          if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
          fi
          MODEL_NAME="${ALTAI_PARAM_MODEL_NAME}" \
          RUN_NAME="${ALTAI_PARAM_RUN_NAME}" \
          MAX_STEPS="${ALTAI_PARAM_MAX_STEPS}" \
          PYTHONNOUSERSITE=1 \
          UV_INDEX_URL="https://pypi.org/simple" \
          UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124" \
          uv run --script train.py 2>&1 | tee "$out/logs/train.log"
    outputs:
      train_log:
        type: file
        path: /outputs/logs/train.log
        role: logs
      verdict:
        type: file
        path: /outputs/metrics/verdict.txt
        role: metrics
```

Container fallback shape:

```yaml
steps:
  - id: train
    execution: container
    image: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
    with:
      command:
        - bash
        - -lc
        - |
          set -euo pipefail
          out="${CINDERFLOW_OUTPUTS:-/outputs}"
          mkdir -p "$out/logs" "$out/metrics"
          printf '%s' '<BASE64_ENCODED_SCRIPT>' | base64 -d > train.py
          python -m py_compile train.py
          python3 -m venv .venv
          PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --upgrade pip
          PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6,<2.8"
          PYTHONNOUSERSITE=1 .venv/bin/python -m pip install "transformers<5" "sentence-transformers[train]" "datasets>=2.18.0,<4" "huggingface-hub>=0.21.2,<1" "fsspec>=2023.12.0,<2025" "pyarrow<19" accelerate scikit-learn
          PYTHONNOUSERSITE=1 .venv/bin/python train.py 2>&1 | tee "$out/logs/train.log"
    outputs:
      train_log:
        type: file
        path: /outputs/logs/train.log
        role: logs
      verdict:
        type: file
        path: /outputs/metrics/verdict.txt
        role: metrics
```

Do not add the model directory as a declared artifact in the first smoke workflow. If model persistence is required, push to Hugging Face Hub/S3 from the script or add a separate explicit production artifact step after the smoke workflow succeeds.

## Script adaptation requirements

The adapted script must:

- Preserve the production template scaffolding. If the template files are missing, stop and ask for them; do not synthesize production training code from memory.
- Accept `MODEL_NAME`, `DATASET_NAME`, `RUN_NAME`, `MAX_STEPS`, `SMOKE_TEST`, `HF_REPO_ID`, and `OUTPUT_DIR` from environment variables or CLI args. Always support `CINDERFLOW_OUTPUTS` for workflow runs.
- Use `MAX_STEPS=1` and a tiny dataset slice for smoke tests.
- Write the final model under `${OUTPUT_DIR}/model`.
- Write logs under `${OUTPUT_DIR}/logs`.
- Write a verdict file under `${OUTPUT_DIR}/metrics/verdict.txt`.
- Print the verdict line to stdout so workflow logs show it.
- Wrap Hub push in `try/except`; skip push when `HF_REPO_ID` is empty.

## Output handling

Use Cinderflow artifacts for workflow smoke outputs:

```bash
cinderflow workflow status "$JOB_ID" --json
cinderflow workflow logs "$JOB_ID" --tail 400
cinderflow artifacts ls "$JOB_ID" --json
cinderflow artifacts download --job "$JOB_ID" --artifact train.verdict --output ./verdict.txt
cinderflow artifacts download --job "$JOB_ID" --artifact train.train_log --output ./train.log
```

Do not report success unless logs contain `VERDICT:`, artifacts include `train.verdict` and `train.train_log`, and `verdict.txt` contains a valid verdict line. If logs show default container command/no outputs, the workflow did not run training even if state is `SUCCEEDED`.

Use `cinderflow files` only for debugging workspace files or one-off small files. For larger model directories, prefer Hugging Face Hub/S3 or an explicit production artifact strategy after the smoke workflow succeeds.

## Failure handling

- If workflow validation fails, stop and fix the YAML. Do not submit.
- If workflow execution fails, collect `workflow status`, `workflow logs --tail 400`, and artifacts if any; then clean up the GPU. Do not call a workflow failed only because setup is slow or operation polling has not changed yet.
- If an `exec` diagnostic fails with exit code `255` or `timed out`, run the diagnostic commands above before blaming Python, pip, sentence-transformers, or Cinderflow server instability.
- Always remove paid ThunderCompute GPUs unless the user explicitly asks to keep them.
