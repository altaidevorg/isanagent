# LlamaFactory Workflow

## Purpose
Define the shared LlamaFactory workflow for training: installation, environment checks, execution boundary, and failure handling.

## Shared rules
- Use `hiyouga/LlamaFactory` for all training work.
- Do not replace it with another training framework or a custom training loop.
- If `LlamaFactory` is missing or broken, run `scripts/install_llamafactory.sh` first.
- When training fails, stay inside the LlamaFactory workflow and debug the real prerequisite, config, data, or resource issue.
- Leave the final result ready for downstream evaluation.

## Prerequisite checks
Before running training, confirm the following:
- `LlamaFactory` is installed and runnable.
- The relevant script under `scripts/` is present.
- A valid base model path or model id is available.
- The current Python environment and required dependencies are usable.
- The required training inputs for the selected mode are available.

## Config discipline
Before running LlamaFactory, inspect the training config instead of relying on defaults or memory.

Pay special attention to parameters that strongly affect training behavior:
- `model_name_or_path`: use the intended base model or checkpoint.
- `finetuning_type`: keep it consistent with the selected stage; do not switch to LoRA when full-parameter fine-tuning is required.
- `template`: must match the benchmark-facing model input format used by the evaluation path; do not use a default or a random template without verifying this match.
- `dataset` and `dataset_dir`: point to the current prepared training data.
- `learning_rate`, `num_train_epochs` or `max_steps`, `per_device_train_batch_size`, and `gradient_accumulation_steps`: keep the training strength deliberate.
- `cutoff_len`: avoid truncating important prompt, reasoning, or answer content.

If an existing workspace config is available, start from it and change only what the current run requires.

## Failure handling
- If the failure indicates that `LlamaFactory` or its CLI is missing or broken, run `scripts/install_llamafactory.sh` first.
- If LlamaFactory fails with a dependency version check such as `transformers>=...,<=... is required`, use `DISABLE_VERSION_CHECK=1` for the retry. The provided `scripts/run_llamafactory.sh` already sets this for training runs.
- If the failure is caused by data, config, or resource issues, fix those issues without switching frameworks.
- After fixing a failure, rerun a small validation job before continuing the intended run.

## Output expectation
- Training should remain reproducible within the current workspace setup.
- The result should be exported as `final_model/` under the project root (`train/projects/{project_id}/final_model/`).
- `final_model/` should be loadable by common inference stacks (e.g. vLLM) when applicable.
- The exported model should be left ready for real evaluation rather than as an unfinished intermediate artifact.
