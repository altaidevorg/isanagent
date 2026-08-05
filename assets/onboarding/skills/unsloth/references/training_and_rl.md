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

> [!NOTE]
> **TRL Parameter Compatibility**: In TRL v0.15+, `GRPOTrainer` uses `processing_class=tokenizer` instead of `tokenizer=tokenizer`.

### Step-by-Step GRPO Workflow

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "unsloth_zoo",
#     "trl",
#     "transformers",
#     "peft",
#     "datasets",
#     "accelerate",
#     "torch",
#     "vllm",
# ]
# ///

import os
# Disable FlashInfer on Tesla T4 / Turing GPUs (compute capability 7.5)
os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"

import re
import torch
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig

# 1. Patch TRL GRPOTrainer with Unsloth Fused Kernels
PatchFastRL("GRPO", FastLanguageModel)

# 2. Load Model & Tokenizer with Fast vLLM Inference & VRAM Allocation Limit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Math-7B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,       # Boot vLLM sampling engine
    gpu_memory_utilization = 0.4, # Reserve VRAM for training gradients
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    use_gradient_checkpointing = "unsloth",
)

# 3. Define Group Reward Functions
def xml_layout_reward_func(completions, **kwargs):
    """Reward completion if it follows <think>...</think><answer>...</answer> structure."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        rewards.append(1.0 if re.match(pattern, text, re.DOTALL) else 0.0)
    return rewards

def math_answer_reward_func(prompts, completions, answer, **kwargs):
    """Reward completion if extracted answer matches ground truth."""
    rewards = []
    for completion, target in zip(completions, answer):
        text = completion[0]["content"]
        extracted = text.split("<answer>")[-1].split("</answer>")[0].strip() if "<answer>" in text else ""
        rewards.append(2.0 if extracted == str(target).strip() else 0.0)
    return rewards

# 4. Configure GRPO Trainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer, # TRL v0.15+ parameter name
    reward_funcs = [xml_layout_reward_func, math_answer_reward_func],
    args = GRPOConfig(
        use_vllm = True,
        vllm_gpu_memory_utilization = 0.4, # Limit vLLM VRAM usage to prevent OOM
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = True,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_generations = 8,          # Number of completion samples per prompt
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
from unsloth import FastLanguageModel, PatchFastRL
from trl import DPOTrainer, DPOConfig

PatchFastRL("DPO", FastLanguageModel)

trainer = DPOTrainer(
    model = model,
    ref_model = None, # Unsloth automatically handles reference model implicitly
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        beta = 0.1, # DPO temperature scaling
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs/DPO",
    ),
)
trainer.train()
```

---

## 4. VRAM Reduction Checklist

If encountering CUDA Out-Of-Memory (OOM) errors during training:
1. Ensure `load_in_4bit = True` is set.
2. Set `use_gradient_checkpointing = "unsloth"` in `get_peft_model`.
3. Set `optim = "adamw_8bit"` or `"paged_adamw_8bit"`.
4. Set `per_device_train_batch_size = 1` and increase `gradient_accumulation_steps`.
5. Set `lora_dropout = 0.0` to enable Triton fused path.
6. Enable `packing = True` in `SFTTrainer` to reduce padding token overhead.
7. Limit vLLM memory: `gpu_memory_utilization = 0.4` and `vllm_gpu_memory_utilization = 0.4`.
