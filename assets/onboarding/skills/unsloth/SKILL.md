---
name: unsloth
description: Operational reference and helper scripts for using the Unsloth library for LLM, VLM, Tool-Calling, and Sentence Transformer fine-tuning, RL (GRPO/DPO), preflight diagnostics, dataset auditing, run planning, and GGUF/Ollama export.
---

# Unsloth Agent Skill

This skill provides concise guidelines, reference documentation, preflight verification tools, and production-ready Python scripts for fine-tuning, evaluating, and exporting Large Language Models (LLMs), Vision-Language Models (VLMs), Tool-Calling Agents, Sentence Transformers, and Reinforcement Learning models (GRPO/DPO) using the **Unsloth** library across CUDA, ROCm, XPU, and Apple Silicon (MLX) backends.

---

## Agent Operational Decision Flow

Agents executing fine-tuning tasks MUST follow this step-by-step operational workflow:

```
1. Run Environment Doctor ──► 2. Audit Dataset ──► 3. Generate Run Plan ──► 4. Fine-Tune ──► 5. Verify & Export
   (scripts/doctor.py)          (scripts/audit_dataset.py)  (scripts/plan_run.py)       (scripts/finetune_*.py)  (scripts/export_gguf_ollama.py)
```

### Step 1: Preflight Environment Inspection
Run `scripts/doctor.py` to inspect OS, accelerator hardware, VRAM, package versions, and bfloat16 support before configuring fine-tuning.

### Step 2: Dataset Audit
Run `scripts/audit_dataset.py --max_seq_length 4096` to check dataset schema compliance, turn integrity, token length percentiles, and truncation rate.

### Step 3: Run Planning & Calibration
Run `scripts/plan_run.py --workflow <sft|dpo|grpo> --vram_gb <16|24|80>` to calculate batch sizes, gradient accumulation steps, and verify GRPO divisibility (`effective_batch_size % num_generations == 0`).

### Step 4: Execute Fine-Tuning Script
Copy and run the appropriate workflow script with `--help` flag check first:
- **Supervised Fine-Tuning**: Run `scripts/finetune_sft.py --help`
- **Tool-Calling Agent Tuning**: Run `scripts/finetune_tool_calling.py --help`
- **GRPO Reasoning RL**: Run `scripts/finetune_grpo_reasoning.py --help`
- **DPO Preference Fine-Tuning**: Run `scripts/finetune_dpo.py --help`
- **Vision-Language Model**: Run `scripts/finetune_vision.py --help`
- **Sentence Transformer**: Run `scripts/finetune_embedding.py --help`

### Step 5: Verification & Export
Evaluate model outputs, test reward functions with `scripts/test_rewards.py`, profile performance with `scripts/benchmark_memory_speed.py`, and export GGUF / merged weights with `scripts/export_gguf_ollama.py`.

---

## Core Operational Rules

1. **Inline Script Metadata (PEP 723)**: Standalone scripts contain PEP 723 metadata (`# /// script ... # ///`). Use your environment's appropriate script runner (`uv`, `python`, `cinderflow exec`, etc.) for automatic dependency resolution.
2. **Model Freshness**: Use up-to-date defaults (`Qwen3-8B` for text LLMs, `gemma-4-12b-it`, `Qwen3.6-27B` for 27B+ models). Check Hugging Face Hub for fresh versions unless a specific model is explicitly requested.
3. **Native Chat Templates**: Prioritize `tokenizer.apply_chat_template(...)` directly out of the box. Use `get_chat_template()` only as a fallback when native chat templates are missing.
4. **LoRA Dropout**: `lora_dropout=0.0` is recommended for Triton fused kernel acceleration and VRAM savings, but non-zero dropout (e.g. `0.05`) is supported and recommended when overfitting occurs.
5. **Sequence Packing Gotchas**: Enable `packing=True` only when Flash Attention variable-length kernels (`flash_attn_varlen_func`) are active to prevent cross-sample attention leakage.

---

## Technical Reference Guides (`references/`)

* 📖 [Model Loading & PEFT Guide](references/model_loading_and_peft.md): Quantization (4-bit NF4, FP8), LoRA/QLoRA configuration, Zero Dropout rules, RoPE scaling, and MLX Apple Silicon support.
* 📖 [Datasets & Chat Templates Guide](references/dataset_and_chat_templates.md): Native chat templates vs `get_chat_template()`, ShareGPT standardization, response-only loss masking, and raw text chunking.
* 📖 [Training & RL (GRPO/DPO) Guide](references/training_and_rl.md): Supervised fine-tuning (`SFTTrainer`), Reasoning RL (`GRPOTrainer` + vLLM), and preference fine-tuning (`DPOTrainer` with implicit reference models).
* 📖 [Sequence Packing & Prefix Caching Guide](references/sequence_packing_and_prefix_caching.md): Sequence packing mechanics, Flash Attention varlen masking, multi-epoch trade-offs, and GRPO prefix caching (`PrefixGrouper`).
* 📖 [Saving, Export & Deployment Guide](references/saving_export_and_deployment.md): Merging LoRA adapters, local GGUF quantization export, pushing GGUF + Ollama Modelfile to HF Hub, and fast vLLM decoding.
* 📖 [CLI, Platform & `uv` Hardware Matrix](references/cli_and_platform_matrix.md): `uv` environment setup, Colab auto-install headers, multi-GPU `torchrun` execution, and hardware compatibility rules (Ampere, Hopper, Turing Tesla T4, Apple Silicon).
* 📖 [Advanced Topics & Architectures Guide](references/advanced_topics_and_architectures.md): Vision-Language Models (VLM), Sentence Transformer / Embedding fine-tuning, and full-parameter tuning.
* 📖 [Troubleshooting & Gotchas Guide](references/troubleshooting_and_gotchas.md): Precision overflow (`FORCE_FLOAT32`), dynamic CUDA linker setup, import order rules, DDP multi-GPU placement, and OOM recovery recipes.

---

## Production Helper Scripts (`scripts/`)

* 🩺 [Preflight Doctor Tool](scripts/doctor.py): Environment, GPU accelerator, VRAM, and package matrix diagnostic script.
* 📊 [Dataset Auditor Tool](scripts/audit_dataset.py): Dataset turn integrity, token length percentiles, and truncation inspector.
* 📋 [Training Run Planner Tool](scripts/plan_run.py): Execution planner calculating batch sizes and enforcing GRPO divisibility rules.
* 🧪 [Reward Function Test Tool](scripts/test_rewards.py): GRPO XML layout and correctness reward function unit test suite.
* 🚀 [Supervised Fine-Tuning Script](scripts/finetune_sft.py): SFT pipeline template with response-only masking and native chat templates.
* 🛠️ [Tool-Calling Agent Fine-Tuning Script](scripts/finetune_tool_calling.py): Agent SFT template with structured function call message formatting.
* 🧠 [GRPO Reasoning RL Script](scripts/finetune_grpo_reasoning.py): DeepSeek R1-style GRPO reasoning RL pipeline with vLLM sampling acceleration.
* ⚖️ [DPO Preference Fine-Tuning Script](scripts/finetune_dpo.py): Direct Preference Optimization with implicit reference model (~5.5GB VRAM saved).
* 👁️ [Vision-Language Model Fine-Tuning Script](scripts/finetune_vision.py): Multimodal VLM fine-tuning pipeline with `FastVisionModel`.
* 🔤 [Sentence Transformer / Embedding Script](scripts/finetune_embedding.py): Contrastive embedding fine-tuning with `FastSentenceTransformer`.
* 📦 [Dataset Sequence Packing Script](scripts/pack_and_preprocess_dataset.py): Dataset preprocessing, chat template formatting, and packing tool.
* 📦 [GGUF & Ollama Export Script](scripts/export_gguf_ollama.py): GGUF quantization, adapter merging, and HF Hub / Ollama pushing.
* ⚡ [Memory & Speed Benchmark Tool](scripts/benchmark_memory_speed.py): VRAM allocation and generation throughput profiling tool.
* 🖥️ [Multi-GPU Distributed Launch Script](scripts/multigpu_sft_launch.sh): `torchrun` multi-GPU wrapper script.
