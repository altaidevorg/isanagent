---
name: unsloth
description: Master Unsloth library for 2-5x faster fine-tuning, 80% memory reduction, model loading (4-bit, 8-bit, 16-bit, FP8, QLoRA, LoRA), RLHF/DPO/GRPO reasoning training, chat template processing, response masking, synthetic data generation, GGUF/Ollama export, and vLLM inference using the ultra-fast `uv` package manager.
---

# Unsloth Master Skill

This skill provides comprehensive operational knowledge for using the **Unsloth** library to fine-tune, optimize, evaluate, and export Large Language Models (LLMs), Vision-Language Models (VLMs), Sentence Transformers / Embedding models, and Reinforcement Learning / Reasoning models (GRPO/DPO) on CUDA, ROCm, XPU, and Apple Silicon (MLX) devices.

---

## Quick Reference Matrix

| Task | Primary API / Class | Key Parameters / Notes |
| :--- | :--- | :--- |
| **Package Manager** | `uv pip install ...` / `uv run ...` | High-speed dependency & script execution via `uv` |
| **Colab Auto-Install**| `uv pip install --system` | Fast `uv` try-except bootstrap block for Colab environments |
| **Model Loading** | `FastLanguageModel.from_pretrained` | `model_name`, `max_seq_length`, `load_in_4bit=True`, `load_in_8bit`, `load_in_fp8`, `dtype` |
| **VLM Loading** | `FastVisionModel.from_pretrained` | `model_name`, `max_seq_length`, `load_in_4bit=True`, `text_only=False` |
| **Embeddings Loading**| `FastSentenceTransformer.from_pretrained` | `model_name`, `max_seq_length`, `load_in_4bit=True` |
| **Diffusion Loading** | `FastDiffusionModel.from_pretrained` | `model_name`, `load_in_4bit=True` |
| **PEFT / LoRA** | `FastLanguageModel.get_peft_model` | `r=16`, `target_modules=[...]`, `lora_alpha=16`, `lora_dropout=0.0` (Zero Dropout) |
| **Chat Formatting**| `get_chat_template` | `tokenizer`, `chat_template="chatml"|"llama-3"|"zephyr"|"gemma"|"qwen-2.5"` |
| **Loss Masking** | `train_on_responses_only` | Mask system/user tokens so loss computes on assistant responses only |
| **SFT Training** | `UnslothTrainer` / `SFTTrainer` | `UnslothTrainingArguments` / `SFTConfig`, `use_gradient_checkpointing="unsloth"` |
| **GRPO Reasoning**| `PatchFastRL`, `GRPOTrainer` | DeepSeek R1-style RL with `vLLMSamplingParams`, `processing_class=tokenizer`, `gpu_memory_utilization=0.4-0.7` |
| **DPO Preferences** | `PatchFastRL`, `DPOTrainer` | Preference fine-tuning with `ref_model=None` (Implicit reference model, ~5.5GB VRAM saved) |
| **Sequence Packing** | `SFTTrainer(..., packing=True)` | Combines short samples to eliminate padding token overhead (100% GPU efficiency) |
| **Model Merging** | `model.save_pretrained_merged` | `save_method="merged_16bit"|"merged_4bit"|"lora"` |
| **GGUF Export** | `model.save_pretrained_gguf` | `quantization_method="q4_k_m"|"q8_0"|"f16"|"q5_k_m"` |
| **Ollama Push** | `model.push_to_hub_gguf` | Quantizes and uploads GGUF alongside Ollama `Modelfile` to Hugging Face Hub |
| **vLLM Inference** | `FastLanguageModel.for_inference` | Enables fast vLLM decoding engine (`fast_inference=True`) |

---

## Core Operational Rules

1. **Package Management (`uv`)**: Always use `uv` for environment setup, package installation, and execution for maximum speed and reproducibility:
```bash
# Environment setup
uv venv .venv --python 3.11 && source .venv/bin/activate
# Install Unsloth & dependencies
uv pip install unsloth unsloth_zoo
# Run scripts using uv
uv run python scripts/finetune_sft.py
```
2. **Google Colab Automated Bootstrapping with `uv`**: In Google Colab notebooks, include an automated `uv` try-except bootstrap header at the very top of scripts (`uv pip install --system ...`) to install Unsloth automatically in 10-15 seconds.
3. **Top Import Rule**: ALWAYS import `unsloth` before `transformers`, `peft`, or `trl`.
```python
import unsloth
from unsloth import FastLanguageModel, FastVisionModel, FastModel
```
4. **Zero Dropout Requirement**: Keep `lora_dropout = 0.0` in `get_peft_model`. Unsloth's Triton custom fused kernels (which eliminate intermediate activation caching and save ~80% VRAM) require `lora_dropout = 0.0`.
5. **Gradient Checkpointing**: Use `use_gradient_checkpointing="unsloth"` (or `"unsloth_offload"`) in `get_peft_model` or `TrainingArguments` to save up to 80% VRAM.
6. **Completion-Only Training**: Always apply `train_on_responses_only` or pass `completion_only_loss` to prevent models from learning to predict system/user prompts.
7. **Implicit Reference Model in DPO**: Always set `ref_model = None` in `DPOTrainer`. Unsloth computes reference logits on-the-fly, saving an entire copy of the 7B model in VRAM (~5.5 GB saved).
8. **Fast Inference Mode**: Before running generation loops, execute `FastLanguageModel.for_inference(model)` to double inference throughput.
9. **Backend Safety**: On Apple Silicon (macOS), Unsloth automatically uses the MLX backend (`unsloth._IS_MLX`). SFT training and GGUF export run on MLX; GRPO/DPO RL requires CUDA/ROCm GPUs.
10. **Hardware-Aware FlashInfer Compatibility**: Disable FlashInfer (`os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"`) **ONLY** on older GPUs (Tesla T4 / Turing architecture, compute capability $\le 7.5$). On modern Ampere/Hopper/Ada GPUs (A100, H100, L4, RTX 3090/4090, compute capability $\ge 8.0$), keep FlashInfer enabled for maximum speed.
11. **Dynamic VRAM Allocation**: On VRAM-constrained GPUs (e.g. 16GB Tesla T4 / RTX 4060Ti), set `gpu_memory_utilization=0.4` and `vllm_gpu_memory_utilization=0.4` to prevent GRPO OOM errors. On high-VRAM GPUs (24GB, 40GB, 80GB A100/H100), scale this value up to `0.6`–`0.8`.

---

## Response Masking Cheat Sheet

When calling `train_on_responses_only(trainer, instruction_part=..., response_part=...)`, use these exact delimiters:

- **ChatML / Qwen 2.5**:
  - `instruction_part = "<|im_start|>user\n"`
  - `response_part = "<|im_start|>assistant\n"`
- **Llama 3 / 3.1 / 3.2 / 3.3**:
  - `instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"`
  - `response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"`
- **Gemma 2**:
  - `instruction_part = "<start_of_turn>user\n"`
  - `response_part = "<start_of_turn>model\n"`

---

## Standard Workflows & Code Patterns

### 1. Supervised Fine-Tuning (SFT) - Causal LLM
```python
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# 1. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length = 4096,
    dtype = None, # Auto-detect (bf16/fp16)
    load_in_4bit = True,
)

# 2. Add LoRA Adapters (Zero dropout enables fused Triton kernels)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.0, # Zero dropout required for Triton fused path
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Apply Chat Template
tokenizer = get_chat_template(tokenizer, chat_template = "qwen-2.5")

# 4. Prepare Dataset with Response-Only Masking
def formatting_prompts_func(examples):
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in examples["messages"]]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. Train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 4096,
    dataset_num_proc = 2,
    packing = False, # Set True for fast text packing
    args = SFTConfig(
        per_device_train_batch_size = 1, # 1 for 16GB VRAM, increase gradient_accumulation_steps
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer = train_on_responses_only(trainer, instruction_part = "<|im_start|>user\n", response_part = "<|im_start|>assistant\n")
trainer.train()
```

### 2. Reinforcement Learning (GRPO Reasoning - DeepSeek R1 Style)
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
import re
import sys
import torch

# Dynamic FlashInfer Check: Disable FlashInfer ONLY on older GPUs (Turing / Tesla T4 compute capability <= 7.5)
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    if major < 8:  # Tesla T4, GTX 1660, RTX 2080
        os.environ.setdefault("UNSLOTH_VLLM_NO_FLASHINFER", "1")

from unsloth import FastLanguageModel, PatchFastRL
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# Enable Unsloth Fast RL & vLLM acceleration
PatchFastRL("GRPO", FastLanguageModel)

# Determine memory allocation based on available VRAM
gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 16
vllm_mem_util = 0.4 if gpu_vram_gb <= 16 else 0.7

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Math-7B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,              # Boot vLLM sampling engine
    gpu_memory_utilization = vllm_mem_util, # Dynamic VRAM allocation
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.0, # Zero dropout required for Triton fused path
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# Define Reward Functions
def xml_layout_reward_func(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        rewards.append(1.0 if re.match(pattern, text, re.DOTALL) else 0.0)
    return rewards

def correctness_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, target in zip(completions, answer):
        text = completion[0]["content"]
        extracted = text.split("<answer>")[-1].split("</answer>")[0].strip() if "<answer>" in text else ""
        rewards.append(2.0 if extracted == str(target).strip() else 0.0)
    return rewards

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [xml_layout_reward_func, correctness_reward_func],
    args = GRPOConfig(
        use_vllm = True,
        vllm_gpu_memory_utilization = vllm_mem_util,
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
        num_generations = 8,
        max_prompt_length = 256,
        max_completion_length = 512,
        output_dir = "outputs/GRPO",
    ),
    train_dataset = dataset,
)
trainer.train()
```

### 3. GGUF Export & Ollama Publishing
```python
# Save local GGUF (e.g. Q4_K_M)
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")

# Push GGUF directly to Hugging Face Hub (includes Ollama Modelfile auto-generation)
model.push_to_hub_gguf(
    "username/my-custom-model-gguf",
    tokenizer,
    quantization_method = ["q4_k_m", "q8_0"],
    token = "hf_...",
)

# Run with Ollama in terminal:
# ollama run hf.co/username/my-custom-model-gguf:Q4_K_M
```

---

## Detailed Skill References & Scripts

For deep-dive technical guides and production-ready executable scripts, refer to:

### Technical Reference Guides (`references/`)
- 📖 [Model Loading & PEFT Guide](references/model_loading_and_peft.md)
- 📖 [Datasets & Chat Templates Guide](references/dataset_and_chat_templates.md)
- 📖 [Training & RL (GRPO/DPO) Guide](references/training_and_rl.md)
- 📖 [Saving, Export & Deployment Guide](references/saving_export_and_deployment.md)
- 📖 [CLI, Platform & `uv` Hardware Matrix](references/cli_and_platform_matrix.md)
- 📖 [Advanced Topics & Architectures Guide](references/advanced_topics_and_architectures.md)
- 📖 [Troubleshooting, NaN Loss & Gotchas Guide](references/troubleshooting_and_gotchas.md)
- 📖 [Triton Fused Kernels & MoE Internals Guide](references/triton_kernels_and_architecture_internals.md)
- 📖 [Sequence Packing & Prefix Caching Guide](references/sequence_packing_and_prefix_caching.md)

### Production Helper Scripts (`scripts/`)
- 🚀 [Supervised Fine-Tuning Script](scripts/finetune_sft.py)
- 🧠 [GRPO Reasoning RL Script](scripts/finetune_grpo_reasoning.py)
- ⚖️ [DPO Preference Fine-Tuning Script](scripts/finetune_dpo.py)
- 👁️ [Vision-Language Model Fine-Tuning Script](scripts/finetune_vision.py)
- 🔤 [Sentence Transformer / Embedding Script](scripts/finetune_embedding.py)
- 📦 [Dataset Sequence Packing Script](scripts/pack_and_preprocess_dataset.py)
- 📦 [GGUF & Ollama Export Script](scripts/export_gguf_ollama.py)
- ⚡ [Memory & Speed Benchmark Tool](scripts/benchmark_memory_speed.py)
- 🖥️ [Multi-GPU Distributed Launch Script](scripts/multigpu_sft_launch.sh)
