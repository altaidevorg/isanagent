---
name: oom-recovery-playbook
description: Out-of-memory and resource pressure recovery without changing the user's training objective.
always: false
---

# OOM recovery playbook

When a run fails with CUDA OOM, host OOM, or timeout:

## Do first (preserves the objective)

1. **Reduce per-step memory** — lower `per_device_train_batch_size` (or equivalent) and **raise** `gradient_accumulation_steps` so **effective** batch size stays the same when possible.
2. **Checkpointing** — enable gradient checkpointing / activation checkpointing if the framework supports it.
3. **Precision** — move to bf16/fp16 only if it does not change the user’s stated precision requirement.
4. **Hardware** — move to a larger GPU or more RAM **before** changing the algorithm.

## Do not do without explicit user consent

- Switching full fine-tuning → LoRA/QLoRA (changes what is trained).
- Silently lowering `max_length` / context window (changes what data the model sees).
- Swapping datasets or model checkpoints “because they fit”.
- Disabling logging or monitoring to hide failures.

## After stabilization

Re-run a **small** pilot, confirm loss/metrics move as expected, then restore full scale.
