# Training and Reinforcement Learning (GRPO / DPO) Reference Guide

This reference provides complete technical instructions for Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), Preference Fine-Tuning (DPO), and DeepSeek R1-style **Group Relative Policy Optimization (GRPO)** in **Unsloth**.

---

## 1. Supervised Fine-Tuning (SFT)

Unsloth patches TRL's `SFTTrainer` and Hugging Face's `TrainingArguments` into `UnslothTrainer` and `UnslothTrainingArguments` (or `SFTConfig`).

### Key Memory Optimization Arguments in `SFTConfig`

| Argument | Value | Purpose |
| :--- | :--- | :--- |
| `optim` | `"adamw_8bit"` / `"paged_adamw_8bit"` | Quantizes AdamW optimizer states to 8-bit, saving 75% optimizer VRAM. |
| `gradient_checkpointing` | `True` | Set `use_gradient_checkpointing="unsloth"` in `get_peft_model` for maximum savings. |
| `per_device_train_batch_size` | `1` to `4` | Small batch size per GPU. Increase `gradient_accumulation_steps` to compensate. |
| `gradient_accumulation_steps` | `4` to `16` | Simulates large effective batch sizes without VRAM spikes. |
| `learning_rate` | `2e-4` (LoRA) / `2e-5` (Full) | Recommended default learning rate for 4-bit QLoRA. |
| `fp16` / `bf16` | `bf16=True` on Ampere+ | Use `bf16` whenever supported for stable gradient norms. |

### Code Example: Standard SFT Setup (TRL Standard Signature)
```python
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",
    max_seq_length = 4096,
    load_in_4bit = True,
)

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,  # TRL standard parameter name
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = 4096,
        packing = False,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        output_dir = "outputs/sft_model",
    ),
)
trainer.train()
```

---

## 2. Reinforcement Learning via GRPO (DeepSeek R1 Reasoning)

Group Relative Policy Optimization (GRPO) eliminates the need for a separate Critic / Value Model by sampling a group ($G$) of outputs per prompt and calculating normalized relative rewards within the group.

### Unsloth Fast RL Integration (`PatchFastRL`)
Unsloth patches TRL's `GRPOTrainer` with `PatchFastRL("GRPO", FastLanguageModel)`:
- Integrates vLLM for ultra-fast generation of candidate completions.
- Replaces logit matrix calculations with fused Triton cross-entropy kernels.
- Supports multi-reward functions (correctness, formatting, xml compliance, code execution).

> [!IMPORTANT]
> **Tesla T4 / Turing GPU Compatibility**: On Tesla T4 GPUs (compute capability 7.5), FlashInfer sampling kernels cause vLLM initialization failures. Disable FlashInfer by setting `os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"` before importing `unsloth`.

> [!IMPORTANT]
> **TRL GRPO Divisibility Rule**: Effective batch size (`per_device_train_batch_size * gradient_accumulation_steps`) MUST be divisible by `num_generations` (e.g. $1 \times 8 = 8 \pmod 8 == 0$).

### Step-by-Step GRPO Workflow

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth>=2026.8.0",
#     "unsloth_zoo>=2026.8.0",
#     "trl>=1.9.2",
#     "transformers>=5.14.1",
#     "peft>=0.20.0",
#     "datasets>=5.0.1",
#     "accelerate>=1.14.0",
#     "torch>=2.13.0",
#     "vllm>=0.26.0",
# ]
# ///

import os
import re
import torch
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOTrainer, GRPOConfig

# 1. Patch TRL GRPOTrainer with Unsloth Fused Kernels
PatchFastRL("GRPO", FastLanguageModel)

# 2. Load Model & Tokenizer with Fast vLLM Inference & VRAM Allocation Limit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",
    max_seq_length = 2048,
    load_in_4bit = True,
    fast_inference = True,       # Boot vLLM sampling engine
    gpu_memory_utilization = 0.4, # Reserve VRAM for training gradients
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# 3. Define Group Reward Functions
def xml_layout_reward_func(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        rewards.append(1.0 if re.match(pattern, text, re.DOTALL) else 0.0)
    return rewards

def math_answer_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, target in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        extracted = text.split("<answer>")[-1].split("</answer>")[0].strip() if "<answer>" in text else ""
        rewards.append(2.0 if extracted == str(target).strip() else 0.0)
    return rewards

# 4. Configure GRPO Trainer (Effective batch 1 * 8 = 8 is divisible by num_generations = 8)
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [xml_layout_reward_func, math_answer_reward_func],
    args = GRPOConfig(
        use_vllm = True,
        vllm_gpu_memory_utilization = 0.4,
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,  # 1 * 8 = 8 % 8 == 0
        num_generations = 8,
        max_prompt_length = 256,
        max_completion_length = 512,
        output_dir = "outputs/GRPO",
    ),
    train_dataset = dataset,
)

trainer.train()
```

---

## 3. Direct Preference Optimization (DPO)

DPO optimizes preferences directly on prompt-chosen-rejected pairs without needing a reward model.

```python
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import DPOTrainer, DPOConfig

PatchFastRL("DPO", FastLanguageModel)

trainer = DPOTrainer(
    model = model,
    ref_model = None, # Unsloth automatically handles reference model implicitly (~5.5GB VRAM saved)
    processing_class = tokenizer,
    train_dataset = dataset,
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        beta = 0.1,
        max_prompt_length = 512,
        max_length = 2048,
        learning_rate = 5e-6,
        logging_steps = 1,
        optim = "adamw_8bit",
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        output_dir = "outputs/dpo_model",
    ),
)
trainer.train()
```
