---
name: cinderflow
description: Use when an agent must operate Cinderflow from the public client side: validate and submit workflow YAMLs to configured GPUs, poll operations and job status, follow SSE logs, inspect artifacts, use remote console and Jupyter-backed file commands, provision or remove client-visible GPUs such as ThunderCompute/RunPod/SSH, and safely clean up paid cloud GPUs through an already-running Cinderflow API server. This skill is for remote client operations and troubleshooting API/timeouts/lifecycle issues, not for private server-admin work.
---

# Cinderflow Client Operator

This memory is for an agent that only has access to the public Cinderflow client.
The server is private and already running elsewhere. The agent should know only
the `cinderflow` CLI and the remote API contract exposed through it.

## Mental Model

- Use only the public `cinderflow` CLI.
- Assume `cinderflow-api` is already running and provider credentials live on the
  server.
- Do not use `cinderflowctl`.
- Do not import or reference server/private Python packages.
- Do not ask for provider secrets, SSH keys, Jupyter tokens, database URLs, or
  server internals unless the user explicitly switches to server-admin work.
- Direct `cinderflow` should be the split public client package.
- If direct `cinderflow` is unavailable or old, ask the user to install/update the
  public client package.

## Hard Stop Rules

These rules override shorter examples elsewhere in this file:

- Never use `--type` with `cinderflow gpu configure`; it is not a valid argument.
- Never use `--vcpu`; Cinderflow uses `--cpu-cores`.
- Never run a minimal ThunderCompute configure command such as:

```bash
cinderflow gpu configure --provider thundercompute --gpu-type a6000 --json
```

- For ThunderCompute A6000, use the full safe command:

```bash
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
```

- If a ThunderCompute configure attempt returns `Internal Server Error`, do not
  retry with random GPU types or aliases. Check `data.error` details if present,
  then retry only with the full safe command above.
- If a configure attempt fails or times out, run `cinderflow gpu ls --json`
  before retrying because a paid GPU may already have been created.
- If ThunderCompute logs mention `instance not ready`,
  `Unable to connect to port`, `not ready for GPU probe yet`, or SSH probe
  timeouts, treat this as normal startup latency. Do not start another
  `gpu configure`; wait 30-60 seconds, then run `cinderflow gpu ls --json`.
- Never run multiple ThunderCompute `gpu configure` attempts in parallel.
- For smoke tests, keep at most one active ThunderCompute GPU at a time. If
  multiple active ThunderCompute GPUs exist unexpectedly, stop and clean up the
  extras before workflow, files, or console tests.
- Do not use `cinderflow be` or any oneshot/create-and-submit flow with
  ThunderCompute. Provision first with `gpu configure`, then submit with
  `workflow submit --gpu "$GPU_ID"`.
- If an operation is `FAILED`, treat the operation as the source of truth even
  if `workflow status` still shows a stale `PENDING` or `RUNNING` job state.
- Do not claim Cinderflow lacks monitoring or log following. The client already
  supports operation polling, job status polling, manual sync, and SSE log
  following through `workflow logs --follow`.
- Do not propose a custom Python monitor, retry wrapper, or direct API script
  unless the user explicitly asks for external automation. Prefer the existing
  `cinderflow` commands listed below.

## Required Client Environment

Required:

```bash
CINDERFLOW_API_BASE_URL=http://localhost:8070
CINDERFLOW_API_TOKEN=<jwt>
```

Rules:

- Do not append `/v1` to `CINDERFLOW_API_BASE_URL`.
- Never print full JWTs or secrets.
- If the API is unreachable, report `Connection refused` or timeout as an API
  availability/base URL problem.
- Prefer `--json` for commands the agent parses.
- Always inspect `data.error` before reading success fields.

Health and capability checks:

```bash
cinderflow --help
curl -sS "$CINDERFLOW_API_BASE_URL/v1/health"
curl -sS "$CINDERFLOW_API_BASE_URL/v1/capabilities"
```

Check capabilities before relying on newer features:

- `console`
- `files`
- `sse_logs`
- `artifacts`

If a capability is missing, report it. Do not synthesize unsupported behavior
through server internals.

## JSON Envelope

Cinderflow JSON output shape:

```json
{"command":"...","version":1,"data":{}}
```

Rules:

- If `data.error` exists, the command failed.
- Useful API error codes may be nested under
  `data.error.details.detail.data.error.code`.
- Stable success fields include `operation_id`, `resource_id`, `job_id`, `gpus`,
  `state`, and `artifact_registry`.

## Core Commands

```bash
cinderflow gpu ls --json
cinderflow gpu status --provider <provider> --json
cinderflow gpu discover --provider <provider> --sort hourly_asc --limit 10 --json
cinderflow gpu configure ... --json
cinderflow gpu rm <gpu_id> --json

cinderflow workflow validate <workflow.yaml> --json
cinderflow workflow submit <workflow.yaml> --gpu <gpu_id> --inputs '<json>' --json
cinderflow operations wait <operation_id> --timeout 1800 --poll-interval 30 --json
cinderflow workflow status <job_id> --json
cinderflow workflow logs <job_id> --tail 400
cinderflow workflow logs <job_id> --tail 400 --follow
cinderflow workflow sync <job_id> --json

cinderflow exec --gpu "$GPU_ID" 'echo ok'
cinderflow console --gpu "$GPU_ID"

cinderflow files ls --gpu "$GPU_ID" .
cinderflow files upload --gpu "$GPU_ID" <local_path> <remote_path> [--overwrite]
cinderflow files download --gpu "$GPU_ID" <remote_path> <local_path>
cinderflow files rm --gpu "$GPU_ID" <remote_path> [--recursive]
cinderflow files edit --gpu "$GPU_ID" <remote_path>

cinderflow artifacts ls "$JOB_ID" --json
cinderflow artifacts download --job "$JOB_ID" --artifact <artifact_ref> --output <local_path>
cinderflow artifacts get-url --job "$JOB_ID" --artifact <artifact_ref> --json
```

## Monitoring And Existing Streaming

Cinderflow already has built-in monitoring primitives. Use them before inventing
new scripts.

For an async operation:

```bash
cinderflow operations wait "$OP_ID" --timeout 1800 --poll-interval 30 --json
```

Prefer `operations wait` after `workflow submit`. It waits for the operation and,
once a job id exists, waits for the linked job to reach a terminal state. Use
manual `operations get` polling only when diagnosing wait behavior.

After `workflow submit`, do not replace `operations wait` with a tight loop of
`operations get`. A `RUNNING` operation with no job id can be normal while
Cinderflow deploys utilities, installs dependencies, writes manifest/state, and
starts the remote runner on the GPU.

For a job:

```bash
cinderflow workflow status "$JOB_ID" --json
cinderflow workflow logs "$JOB_ID" --tail 400
cinderflow workflow logs "$JOB_ID" --tail 400 --follow
```

If status or logs appear stale:

```bash
cinderflow workflow sync "$JOB_ID" --json
```

Rules:

- After `workflow submit`, run `operations wait "$OP_ID" --timeout 1800
  --poll-interval 30 --json` as the primary monitor. Do not poll
  `operations get` repeatedly at short intervals as a substitute.
- Never remove a paid GPU while the submitted operation is still `RUNNING`,
  `PENDING`, or missing a terminal result, unless the user explicitly asks to
  abort the run.
- Never remove a paid GPU while the linked job is `PENDING` or `RUNNING`.
  Remote setup can take several minutes before useful logs appear.
- `--follow` uses the server log streaming path; it is the preferred real-time
  workflow log view.
- Optional tuning flags such as `--poll-interval` and
  `--remote-sync-interval` may be used only when the user is debugging stream
  latency or stale remote sync behavior.
- Stop polling or following after `SUCCEEDED`, `FAILED`, `CANCELLED`, or
  `CANCELED`.
- For `500`, timeout, or stale-state cases, inspect `data.error` and current
  state before retrying. Do not blindly resubmit jobs or create another paid GPU.
- If `gpu rm` times out, do not assume cleanup failed. Run `gpu ls --json` and,
  if needed, ask for provider/server-side verification before retrying.

Important syntax:

- `cinderflow gpu status` does not accept a positional GPU ID.
- Use `gpu status --provider <provider>` for provider-level status.
- Use `gpu ls --json` to inspect configured GPU records.
- Use `exec --gpu "$GPU_ID" 'echo ok'` to verify one specific GPU over SSH.
- `cinderflow exec` uses remainder parsing. Do not put `--` before the remote
  command.
- Do not put `--json` after the remote command; it becomes part of the remote
  shell command.

Correct:

```bash
cinderflow exec --gpu "$GPU_ID" 'nvidia-smi'
cinderflow exec --gpu "$GPU_ID" 'echo cinderflow-ssh-ok'
```

Wrong:

```bash
cinderflow exec --gpu "$GPU_ID" -- nvidia-smi
cinderflow exec --gpu "$GPU_ID" nvidia-smi --json
```

## GPU Selection And Paid Resource Policy

- Prefer existing active GPUs before creating a paid GPU.
- If the user gives a `gpu_id`, use that exact GPU unless it is deleted or
  unreachable.
- If an active suitable GPU already exists, ask before creating another paid GPU.
- Before creating a paid GPU, get explicit approval unless the user already asked
  to create/rent/start one.
- Do not remove GPUs the agent did not create unless the user explicitly asks.
- Do not remove historical `deleted` GPU records just to clean the list.
- If the agent creates a paid GPU, save the `gpu_id`.
- If the user did not ask to keep it, remove the paid GPU after the task reaches
  a terminal state.
- Before running `gpu rm`, confirm one of these is true:
  - The relevant `operations wait` result is terminal.
  - The linked `workflow status` result is terminal.
  - The user explicitly asked to abort/cancel/stop the run.
- Do not remove the GPU just because operation polling is slow, job id is not
  assigned yet, deployment is still running, logs are not available yet, or the
  operation is still `RUNNING`.

Cleanup:

```bash
cinderflow gpu rm "$GPU_ID" --json
cinderflow gpu ls --json
```

If `gpu rm` times out, do not assume cleanup failed. Run `gpu ls --json` first.
If the API is down during cleanup, report that server/API access is required and
ask the user/admin to stop the paid resource from the server/provider side.

## Provider Configure General Rules

Do not guess short cloud-provider commands. Build configure commands from:

1. `gpu discover` output.
2. Provider-specific requirements.
3. Validation errors from previous attempts.

General rules:

- Run discovery before choosing GPU type.
- Prefer canonical provider values from discovery.
- Treat validation errors as field-specific learning signals.
- If an error says valid values are `[6 8]`, use only those values next time.
- Do not retry the same failing command unchanged.
- Do not switch GPU types randomly when the error is about CPU, disk, mode,
  missing `gpu_type`, credentials, or SSH readiness.
- Prefer explicit full commands over minimal cloud configure commands.
- Include provider-specific fields explicitly even when CLI defaults exist; CLI
  defaults may be generic and invalid for that provider/GPU combination.
- After every failed `gpu configure`, run `cinderflow gpu ls --json` before
  retrying. The provider may have created a paid instance despite client error or
  timeout.

Troubleshooting pattern:

1. Read `data.error` and nested API details.
2. Identify the failing field: `gpu_type`, `instance_id`, `cpu_cores`,
   `disk_size_gb`, `mode`, `template`, SSH readiness, credentials, etc.
3. Change only that field.
4. Keep known-good fields from the safe command.
5. Check for newly active paid GPUs before retrying.

## Common GPU Configure Arguments

Common arguments and value types:

| Argument | Type | Notes |
| --- | --- | --- |
| `--provider` | string | Required. Examples: `thundercompute`, `runpod`, `ssh`. |
| `--gpu-type` | string | Cloud GPU type. Example: `a6000`. |
| `--gpu-model` | string | Not the same as `--gpu-type`; do not use for ThunderCompute. |
| `--instance-id` | string | Adopt an existing provider instance. |
| `--gpu-index` | integer | Usually `0`. |
| `--cpu-cores` | integer | Provider-specific valid values. Do not use `--vcpu`; Cinderflow uses `--cpu-cores`. |
| `--ram-gb` | integer | Provider-specific. |
| `--disk-size-gb` | integer | Example smoke-test value: `100`. |
| `--mode` | string | Provider-specific. ThunderCompute smoke tests use `prototyping`. |
| `--template` | string | Provider-specific. ThunderCompute smoke tests use `base`. |
| `--ssh-ready-timeout` | integer seconds | Use larger values for fresh cloud machines, e.g. `300`. |
| `--poll-interval-sec` | integer seconds | SSH readiness poll interval, e.g. `5`. |
| `--host` | string | SSH provider host/IP. |
| `--username` | string | SSH username, provider-specific. |
| `--ssh-key-path` | path | Server-side/API-host path if supported. |

Do not invent argument names. For example, `--vcpu` is not a Cinderflow client
argument; use `--cpu-cores`.

## ThunderCompute Notes

ThunderCompute is paid cloud compute. Be conservative.

Discover first:

```bash
cinderflow gpu discover --provider thundercompute --sort hourly_asc --limit 10 --json
```

Safe A6000 smoke-test command:

```bash
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
```

Wrong ThunderCompute commands:

```bash
cinderflow gpu configure --provider thundercompute --gpu-model a6000 --json
cinderflow gpu configure --provider thundercompute --gpu-type a6000 --json
cinderflow gpu configure --provider thundercompute --gpu-type a100 --json
cinderflow gpu configure --provider thundercompute --gpu-type rtxa6000 --json
```

Why wrong:

- ThunderCompute uses `--gpu-type`, not `--gpu-model`.
- A bare `--gpu-type` can fall back to invalid defaults, such as CPU cores `15`.
- A6000 prototyping must include mode, template, CPU cores, disk size, GPU index,
  and SSH readiness settings.
- Do not try random aliases after a server error.
- Use canonical discovery value, usually `a6000`.
- Do not use `cinderflow be` for ThunderCompute auto-create; configure first,
  then submit the workflow with the returned GPU ID.

Observed ThunderCompute constraints:

- A6000 prototyping valid CPU cores observed: `6` or `8`; prefer `6`.
- Do not invent CPU values like `4`, `12`, or `15`.
- Use `--mode prototyping` unless user requests production.
- Use `--template base` unless user requests otherwise.
- Use `--disk-size-gb 100` for smoke tests.
- If API says `Either instance_id or gpu_type is required for thundercompute`,
  the request omitted `--gpu-type` or `--instance-id`.
- If adopting an existing instance, use `--instance-id <id>`.
- If configure times out, run `gpu ls --json`; a new
  `thundercompute-...-0` may already exist.
- If configure or GPU probe appears stuck, wait 30-60 seconds before the next
  check. Fresh ThunderCompute machines can take several minutes before SSH is
  reachable and GPU probing works.
- If `gpu ls --json` shows a new active ThunderCompute GPU after a failed or
  timed-out configure attempt, use that GPU for validation or cleanup. Do not
  create another GPU.
- Provisioning logs with SSH readiness/probe failures are startup latency, not
  permission to open another paid GPU.

After provisioning:

```bash
cinderflow exec --gpu "$GPU_ID" 'echo cinderflow-ssh-ok'
```

After `gpu configure` succeeds:

- Use the `data.gpu_id` returned by that exact command for subsequent workflow, exec, files, and cleanup commands.
- Do not reuse an old `GPU_ID` environment variable.
- Do not pick the first active GPU from `gpu ls` if a configure response returned a new GPU ID.
- If multiple active GPUs exist, ask which one to use unless the task just created one and returned `data.gpu_id`.
- If configure failed or timed out but `gpu ls --json` shows a newly active
  ThunderCompute GPU, use that GPU ID or clean it up. Do not issue another
  configure request first.

ThunderCompute workflow rule:

- Never run `cinderflow be` for ThunderCompute.
- Never send a workflow through a oneshot path that mixes `provider`,
  auto-create defaults, and an already configured ThunderCompute `gpu_id`.
- Correct flow:
  1. `cinderflow gpu configure ...full thundercompute args... --json`
  2. Save returned `data.gpu_id`.
  3. `cinderflow workflow submit <workflow.yaml> --gpu "$GPU_ID" --json`
  4. Run `cinderflow operations wait "$OP_ID" --timeout 1800 --poll-interval 30 --json`.
- If `operations wait` returns `data.operation_status=FAILED`, do not keep
  polling status as if the job is still running. Report the operation error,
  check `gpu ls`, and clean up the paid GPU only if cleanup is allowed by the
  terminal-state rules.

## RunPod Notes

- RunPod can report no available instances for selected GPU types; this is
  provider capacity, not necessarily a Cinderflow client issue.
- Prefer explicit lifecycle commands for paid resources.
- Avoid `cinderflow be` unless user explicitly wants one-shot create/submit.
- If RunPod accidentally creates a pod, clean it up with `cinderflow gpu rm`.

## Workflow Best Practices

When generating workflow YAMLs that need an embedded script, prefer writing the
script from Base64 inside the step command instead of placing a long Python
script directly in a YAML block scalar. This avoids indentation, quoting,
backslash, and brace issues in generated YAML.

Example pattern:

```yaml
steps:
  - id: train
    execution: container
    image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
    with:
      command:
        - bash
        - -lc
        - |
          set -euo pipefail
          out="${CINDERFLOW_OUTPUTS:-/outputs}"
          work="/workspace/run"
          mkdir -p "$work" "$out/logs" "$out/metrics"
          cd "$work"
          printf '%s' '<BASE64_ENCODED_SCRIPT>' | base64 -d > train.py
          python -m py_compile train.py
          python3 -m venv .venv
          PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --upgrade pip
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

For container workflow steps, prefer `python3 -m venv` because standard
framework images do not always include `uv`. Use `uv` for host-level `exec`
diagnostics only when the host already has it. Always use `PYTHONNOUSERSITE=1`
for dependency installation and training commands when isolating from system or
user site packages.

For short generated scripts, prefer inline native execution when Docker image
pull/startup is unnecessary. Inline native does not require a GitHub repo; it
runs in an empty job-local workspace. The command should create its own `uv`
environment. This is the preferred pattern for small Python smoke tests,
including sentence-transformers smoke training:

```yaml
steps:
  - id: train
    execution: native
    with:
      command:
        - bash
        - -lc
        - |
          set -euo pipefail
          out="${CINDERFLOW_OUTPUTS:-/outputs}"
          work="$HOME/.cinderflow/workspaces/default/my-native-run-${CINDERFLOW_JOB_ID:-manual}"
          mkdir -p "$work" "$out/logs" "$out/metrics" "$out/model"
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

Rules for inline native workflows:

- Use `execution: native` without `uses` for generated one-file scripts.
- Use `bash -lc` and create a run-specific workspace under
  `~/.cinderflow/workspaces/default/...`.
- Base64-encode generated Python scripts locally and decode them in the workflow
  step. Do not place large Python heredocs directly in YAML.
- Prefer PEP 723 inline script metadata plus `uv run --script train.py` for generated Python scripts. Use explicit `uv venv` and `uv pip install` only for dependency-resolution debugging.
- Prefix dependency installation and Python execution with `PYTHONNOUSERSITE=1`
  to avoid system/user package leakage.
- Tee the training command to `$CINDERFLOW_OUTPUTS/logs/train.log`.
- Write a durable verdict file under `$CINDERFLOW_OUTPUTS/metrics/verdict.txt`.
- Declare both log and verdict as workflow outputs.
- Validate the generated YAML before submit.

## Sentence-Transformers Smoke Training

For sentence-transformers smoke tests on ThunderCompute, prefer a repo-less
native workflow over `cinderflow exec` loops or container workflows. The tested
pattern is:

1. Generate a small production-style Python training script locally.
2. Base64-encode the script into the workflow YAML.
3. Use `execution: native` without `uses`.
4. Put PEP 723 script metadata at the top of `train.py` and run it with `uv run --script`.
5. Include `torch>=2.6,<2.8`, `transformers<5`, and sentence-transformers dependencies in the script metadata.
6. Provide the PyTorch CUDA wheel index through `UV_EXTRA_INDEX_URL`.
7. Run a one-step smoke train with `PYTHONNOUSERSITE=1`.
8. Tee logs and write `VERDICT: WIN|MARGINAL|REGRESSION ...`.
9. Wait with `operations wait`.
10. Verify artifacts and clean up the paid GPU.

PEP 723 metadata pattern:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.6,<2.8",
#   "transformers<5",
#   "sentence-transformers[train]",
#   "datasets",
#   "accelerate",
#   "scikit-learn",
# ]
# ///
```

Run pattern inside the workflow step:

```bash
PYTHONNOUSERSITE=1 \
UV_INDEX_URL="https://pypi.org/simple" \
UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124" \
uv run --script train.py 2>&1 | tee "$out/logs/train.log"
```

Preflight should live inside `train.py` and run through the same PEP 723 path as
training. Put checks like this near script startup:

```python
import torch, transformers, sentence_transformers
print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("sentence_transformers=", sentence_transformers.__version__)
print("cuda_available=", torch.cuda.is_available())
if tuple(map(int, torch.__version__.split("+", 1)[0].split(".")[:2])) < (2, 6):
    raise RuntimeError("PyTorch >=2.6 required")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
```

Training run command:

```bash
HF_HUB_DISABLE_TELEMETRY=1 \
WANDB_DISABLED=true \
TRACKIO_SPACE_ID= \
DISABLE_TRACKIO=1 \
MODEL_NAME="${ALTAI_PARAM_MODEL_NAME}" \
OUTPUT_DIR="$out" \
PYTHONNOUSERSITE=1 \
UV_INDEX_URL="https://pypi.org/simple" \
UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124" \
uv run --script train.py 2>&1 | tee "$out/logs/train.log"
```

Expected success signals from the tested ThunderCompute A6000 smoke path:

- `Running step train_native via UV-native (no Docker)` appears in runner logs.
- a CUDA-enabled PyTorch >=2.6 version is printed.
- If logs mention `Due to a serious vulnerability issue in torch.load` or `upgrade torch to at least v2.6`, regenerate the workflow with `torch>=2.6,<2.8`; do not retry the same old dependency stack.
- `cuda_available= True` is printed.
- The training script prints a `VERDICT: ...` line.
- `workflow status "$JOB_ID" --json` returns `state: SUCCEEDED`.
- `artifacts ls "$JOB_ID" --json` includes `train_native.train_log` and
  `train_native.verdict`.

Do not use `cinderflow exec` as the default for sentence-transformers smoke
training. `exec` is useful for quick diagnostics such as `python3 --version`,
`uv --version`, or `nvidia-smi`, but it is not the preferred training execution
surface because long installs can hit client request timeouts and are harder to
track than workflow operations.

## Workflow Lifecycle

Use existing repository workflow files. Do not invent inline YAML for smoke
tests. In particular, never submit a legacy `name/jobs` YAML shape; Cinderflow
Altai v1 workflows require `metadata` and `steps`.

Validate first:

```bash
cinderflow workflow validate examples/dummy_workflow.yaml --json
```

Submit:

```bash
cinderflow workflow submit examples/dummy_workflow.yaml \
  --gpu "$GPU_ID" \
  --inputs '{"message":"hello from isanagent"}' \
  --json
```

Wrong dummy workflow behavior:

- Do not create your own `dummy-workflow` YAML.
- Do not submit YAML with top-level `name:` and `jobs:`.
- Do not skip `workflow validate`.
- Do not treat operation creation as workflow success.

Altai v1 input substitution:

- Use double-brace expressions such as `${{ inputs.message }}` and
  `${{ inputs.max_steps }}`.
- Never use single-brace expressions such as `${ inputs.max_steps }`; they can
  be passed through as literal strings and
  cause runtime errors such as
  `ValueError: invalid literal for int() with base 10: '${ inputs.max_steps }'`.
- Before submit, inspect generated YAML for the literal pattern `${ inputs.`
  and fix it before validation/submission.

Submit often returns an operation, not an immediate job ID.

```bash
cinderflow operations wait "$OP_ID" --timeout 1800 --poll-interval 30 --json
```

Use `operations wait` as the primary path. It returns `data.job_id` when the job
is known and `data.job_state` when the linked job has reached a terminal state.
If using `operations get` manually, use `data.resource_id` as `JOB_ID` when
present. If operation is `RUNNING`, do not infer failure and do not delete the
GPU. Switch back to `operations wait` or wait at least 30 seconds before a
single diagnostic `operations get`. If operation is `FAILED`, report the
operation error and stop.

Operation ID versus Job ID:

- `op-...` values are operation IDs. Use them only with
  `cinderflow operations get <operation_id> --json`.
- `job-...` values are job IDs. Use them with `workflow status`, `workflow logs`,
  `artifacts`, and job-related commands.
- Never call `workflow status`, `workflow logs`, or `/jobs/...` endpoints with an
  `op-...` value.
- Do not start `workflow logs --follow` until `operations wait "$OP_ID"` returns
  `data.job_id` or `operations get "$OP_ID"` returns `data.resource_id`.
- If logs are needed after submit, use `data.job_id` from `operations wait`, or
  run one diagnostic `operations get "$OP_ID" --json` and save
  `data.resource_id` as `JOB_ID`; then run
  `cinderflow workflow logs "$JOB_ID" --tail 400 --follow`.

Success criteria:

- `operations wait "$OP_ID" --json` must end with `data.operation_status`
  `SUCCEEDED`.
- `data.job_id` must contain a job id.
- `workflow status "$JOB_ID" --json` must return terminal `SUCCEEDED`.
- If the operation fails with schema validation errors, the workflow did not
  run. Report that the submitted YAML was invalid and clean up any paid GPU.
- If the operation is still `RUNNING` before `operations wait` reaches its
  timeout, the workflow is not terminal. Do not call `gpu rm` yet.

Monitor:

```bash
cinderflow workflow status "$JOB_ID" --json
cinderflow workflow logs "$JOB_ID" --tail 400
cinderflow workflow logs "$JOB_ID" --tail 400 --follow
```

If status/logs look stale:

```bash
cinderflow workflow sync "$JOB_ID" --json
```

## Polling And Terminal States

Terminal states:

- `SUCCEEDED`
- `FAILED`
- `CANCELLED`
- `CANCELED`

Rules:

- Once terminal, stop polling/following.
- After `SUCCEEDED`, collect logs once if needed, then stop.
- After `FAILED`, collect status and logs once, report, then stop unless user
  asks for deeper debugging.
- Do not keep calling `workflow status`, `workflow logs --follow`,
  top-level `logs --follow`, or `gpu status` in a loop after terminal state.
- Do not interpret transient SSH timeout logs after terminal state as job
  failure.
- If the agent created a paid GPU and user did not ask to keep it, remove it.

## Status Timeout Strategy

If `workflow status` or `workflow logs` times out:

1. Prefer waiting on the original operation:
   ```bash
   cinderflow operations wait "$OP_ID" --timeout 1800 --poll-interval 30 --json
   ```
2. If diagnosing manually, check operation once:
   ```bash
   cinderflow operations get "$OP_ID" --json
   ```
3. If `data.job_id` or `data.resource_id` exists, use it as `JOB_ID`.
4. Try one status call:
   ```bash
   cinderflow workflow status "$JOB_ID" --json
   ```
5. If status times out but operation/job is not terminal, wait at least 30
   seconds and retry once. Do not delete the GPU during this window.
6. Try logs once after a `JOB_ID` exists:
   ```bash
   cinderflow workflow logs "$JOB_ID" --tail 400
   ```
7. If terminal state is reached, stop polling.
8. If timeouts persist and no terminal state is known, report that API remote
   sync/SSH is timing out; do not create another GPU and do not remove the
   current GPU unless the user asks to abort.

## SSH Timeout Semantics

- SSH keys, passphrases, provider credentials, and database URLs are
  server/admin concerns. A public client agent should not ask for them or try to
  manage them unless the user explicitly switches to server-admin work.
- Remote runner jobs execute autonomously on the GPU after submit.
- Later status/log/gpu-status commands may trigger remote sync over SSH.
- Server logs like `Failed to connect to remote machine: timed out` can be
  transient sync failures.
- Repeated `Failed to connect to remote machine: timed out` lines usually mean
  some status/log/files/console/sync command is still trying to reach a GPU over
  SSH. Check the nearby HTTP request path in the API logs to identify the caller.
- If operation succeeded and later `workflow status` returns `SUCCEEDED`, the job
  completed despite earlier SSH timeout logs.
- New ThunderCompute machines can be SSH-flaky briefly after provisioning.
- If SSH timeout occurs during configure, check `gpu ls` before retrying.
- If SSH timeout occurs after terminal job state, stop monitoring unless user
  asks for diagnostics.
- If the GPU was already deleted, do not keep polling status, logs, files,
  console, or GPU status for that GPU. Stale checks can keep producing SSH
  timeout noise.
- `Private key file is encrypted` is a server-side SSH key management problem,
  not generic server instability. The server should use its own passphrase-less
  key, prepared by an admin with `cinderflowctl ssh-key ensure`.
- If utility script deployment fails before workflow execution, do not report
  the workflow as healthy just because submit returned an operation ID. Check
  operation/status/logs and report the deployment failure context.

## S3 And External Inputs

- Server and remote runner need credentials. Client-side env alone is not enough
  unless server forwards it.
- Do not paste secrets into workflow YAML or commands.
- Confirm workflow input names before submit.

Zerotune S3 example:

```bash
cinderflow workflow submit examples/multi_step_zerotune_s3.yaml \
  --gpu "$GPU_ID" \
  --inputs '{"adapters_s3_uri":"s3://altai-app-bck/cinderflow/zerotune/e2e-20260406-125145/adapters.tar.gz","dataset_s3_uri":"s3://altai-app-bck/cinderflow/zerotune/e2e-20260406-125145/train.json"}' \
  --json
```

- S3 404 usually means wrong object path or server/runner credentials cannot
  access the object.
- Do not blindly retry the same S3 workflow without changing path or credentials.

## Files And Jupyter Runtime

- `cinderflow files ...` uses server-managed private Jupyter Contents API on the
  remote GPU.
- Client never sees the private Jupyter token or port.
- First `files` or `console` use may install/start Jupyter and take minutes.
- Timeout during first Jupyter use does not always mean GPU is broken; wait and
  retry once.
- Do not call Jupyter directly from client.
- Use relative workspace paths only, e.g. `demo/file.txt`.
- Upload remote paths must include a filename, for example `demo/file.txt`.
  Do not upload to `.`, `demo`, or `demo/` when the intent is to upload a file.
- Do not use absolute paths or `..`. A path like `/remote/test_file.txt` should
  fail with `file_path_outside_workspace` or `403`; this is expected security
  behavior, not server instability.
- `files` paths are relative to the server-managed Jupyter workspace root. Do
  not use `/`, `/outputs`, `/home/...`, or other absolute paths with
  `cinderflow files`.
- `/outputs` is a workflow container step path, not the Jupyter workspace path.
  Use workflow logs/artifacts for step outputs instead of `files ls /outputs`.
- If the user wants workflow outputs, use `workflow logs`,
  `artifacts ls/download`, or the job artifact registry. Do not use
  `cinderflow files` for workflow container output paths.
- Use `files` only for small/medium interactive files.
- Use artifacts/S3/Hugging Face/presigned URLs for large datasets, checkpoints,
  and durable outputs.
- If a relative-path upload times out but later `files ls` shows the file, treat
  the timeout as a transient SSH/Jupyter response issue and verify with
  `files download` or `files ls` before retrying.
- A server log such as `Connection lost ..., reconnecting...` followed by a
  successful `files ls` is not by itself a blocker. Report it as a transient
  reconnect unless file operations continue to fail.
- If `files ls --gpu "$GPU_ID" .` times out, retry once after a short wait and
  then report Jupyter/SSH runtime readiness as uncertain. Do not call it a path
  handling bug when absolute-path requests were the only `403` failures.

File smoke test:

```bash
printf 'hello from cinderflow files\n' > /tmp/cinderflow-file-test.txt
cinderflow files upload --gpu "$GPU_ID" /tmp/cinderflow-file-test.txt demo/cinderflow-file-test.txt --overwrite
cinderflow files ls --gpu "$GPU_ID" demo
cinderflow files download --gpu "$GPU_ID" demo/cinderflow-file-test.txt /tmp/cinderflow-file-downloaded.txt
cmp /tmp/cinderflow-file-test.txt /tmp/cinderflow-file-downloaded.txt
cinderflow files rm --gpu "$GPU_ID" demo/cinderflow-file-test.txt
```

File smoke test rules:

- Every command above must succeed before moving to the next one.
- If upload returns `502`, timeout, or any `data.error`, do not claim download or
  rm passed. Retry once after a short wait, then report the exact failing step.
- If upload fails, do not run download or rm for that remote path until
  `files ls` confirms the file exists.
- If `rm` returns `404`, that only proves the file is absent; it is not a
  successful delete test unless a prior upload/list confirmed the file existed.
- Download is successful only if `cmp` exits `0`.
- Use `--json` when parsing command results, but plain `cmp` is fine for byte
  equality.

## Console

```bash
cinderflow console --gpu "$GPU_ID"
```

- No separate Python REPL command is needed.
- From console, users can run `python`, `ipython`, or `uv run python script.py`.
- Prefer `workflow submit`, `exec`, and `files` unless interactive shell is
  genuinely needed.

## Artifacts Versus Files

- `files`: ad-hoc workspace files for debugging and interactive work.
- workflow artifacts: durable declared job outputs.
- S3/HF/presigned URLs: large datasets, models, checkpoints.
- Do not use `files download` for large artifacts.

Artifact commands:

```bash
cinderflow artifacts ls "$JOB_ID" --json
cinderflow artifacts download --job "$JOB_ID" --artifact <artifact_ref> --output <local_path>
cinderflow artifacts get-url --job "$JOB_ID" --artifact <artifact_ref> --json
```

## Common Errors

- `api_client_config_missing`: set `CINDERFLOW_API_BASE_URL` and
  `CINDERFLOW_API_TOKEN`.
- `token_expired` / `token_revoked`: ask for a fresh token.
- `unauthorized` / `forbidden`: token invalid or missing scopes.
- `Connection refused`: API server is not running or wrong base URL.
- `Either instance_id or gpu_type is required for thundercompute`: add
  `--gpu-type <canonical>` or `--instance-id <id>`.
- `validation: invalid vCPU count ... valid options: [6 8]`: use
  `--cpu-cores 6` or `--cpu-cores 8`; prefer `6`.
- `remote_request_failed: timed out` during configure: run `gpu ls` before
  retrying.
- `Failed to connect to remote machine: timed out`: wait briefly, verify with
  `gpu ls` or `exec`, retry once only if job is not terminal.
- `Internal Server Error` during provider configure: inspect nested error if
  available, then apply provider-specific validation rules. Do not random-retry.
- Stale job state: run `workflow sync <job_id> --json`.

## Agent Safety Checklist

Before paid GPU creation:

1. Check API health.
2. Check capabilities if using newer features.
3. Run `gpu ls`.
4. Run provider discovery.
5. Choose cheapest suitable GPU.
6. Use explicit provider-safe configure command.
7. Confirm user approval if task did not already request paid GPU creation.

After paid GPU creation:

1. Save `gpu_id`.
2. Verify with `exec`.
3. Run task.
4. Stop polling after terminal job state.
5. Remove paid GPU unless user asked to keep it.
6. Verify no unexpected active paid GPUs remain with `gpu ls --json`.
