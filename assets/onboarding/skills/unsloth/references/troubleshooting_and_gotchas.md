# Unsloth Comprehensive Troubleshooting, Gotchas, and Best Practices Guide

This guide documents common operational failure modes, NaN loss diagnosis, precision rules, CUDA linker fixes, vLLM memory conflicts, Tesla T4 FlashInfer gotchas, DDP multi-GPU traps, 16GB VRAM optimization recipes, sequence packing gotchas, chat template selection, and debugging strategies in **Unsloth**.

---

## 1. Mismatched Precision & NaN Loss Prevention (`FORCE_FLOAT32`)

### Symptom: Training loss turns into `NaN` or loss spikes to `inf` at step 1.

#### Cause 1: `float16` precision overflow on newer architectures
Architectures such as Gemma 3, Qwen 3.5, GLM-4 MoE, Gemma 4, and GPT-OSS contain layers (such as gated-deltanet linear attention or high-scale RMSNorm) where standard `float16` dynamic range causes underflow or overflow in loss accumulators.

#### Fix:
1. Always use `bf16 = True` (bfloat16) on Ampere (RTX 3090, A100), Hopper (H100), or Ada GPUs.
2. Unsloth automatically forces float32 accumulator precision for sensitive model types via `FORCE_FLOAT32`. Never override `dtype = torch.float16` on models listed in `FORCE_FLOAT32`.

```python
from unsloth import is_bfloat16_supported

# Standard precision selector in TrainingArguments / SFTConfig:
fp16 = not is_bfloat16_supported()
bf16 = is_bfloat16_supported()
```

---

## 2. Dynamic Linker & CUDA Linking (`bitsandbytes` / `triton`)

### Symptom: `bitsandbytes` error: `CUDA setup failed` or `libcuda.so: cannot open shared object file`.

#### Cause:
In containerized environments (Colab, RunPod, Kaggle, Docker), `/usr/lib64-nvidia` or `/usr/local/cuda` paths are not automatically registered in system `ldconfig`.

#### Fix:
Run the system linker configuration command in terminal before launching Python:

```bash
sudo ldconfig /usr/lib64-nvidia
# Or for CUDA 12.x:
sudo ldconfig /usr/local/cuda-12.4
```

Or verify `bitsandbytes` setup with `uv`:
```bash
uv run python -m bitsandbytes
```

---

## 3. Import Order Warning (`WARNING: Unsloth should be imported before [...]`)

### Symptom: Warning appears at top of script execution:
`WARNING: Unsloth should be imported before [trl, transformers, peft]...`

#### Cause:
If `transformers`, `peft`, or `trl` are imported before `import unsloth`, standard Hugging Face classes instantiate without Unsloth's Triton kernel monkey-patches, causing slower execution and double VRAM usage.

#### Fix:
Ensure `import unsloth` or `from unsloth import ...` is the **very first import statement** in your Python script:

```python
# CORRECT:
import unsloth
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer

# INCORRECT (DO NOT DO THIS):
import torch
from trl import SFTTrainer
import unsloth  # <-- TOO LATE!
```

---

## 4. Multi-GPU (`torchrun` / DDP) Quantized Model Placement

### Symptom: Accelerate device relocation error when running 4-bit / 8-bit models with `torchrun` across multiple GPUs.

#### Cause:
Accelerate attempts to re-map `Linear4bit` parameters across GPU ranks, breaking bitsandbytes pointers.

#### Fix:
Let Unsloth's `prepare_device_map()` assign local ranks automatically:

```python
# Launch via torchrun:
# uv run torchrun --nproc_per_node=2 scripts/finetune_sft.py

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-9B-Instruct",
    device_map = "sequential", # Unsloth automatically remaps to local rank in DDP
    load_in_4bit = True,
)
```

---

## 5. Gradient Checkpointing & Input Require Grads Error

### Symptom: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.

#### Cause:
Input token embeddings were not enabled for gradient computation prior to starting backward pass.

#### Fix:
Use `FastLanguageModel.get_peft_model(..., use_gradient_checkpointing = "unsloth")`. Unsloth automatically calls `enable_input_require_grads()`.

---

## 6. Sequence Packing Cross-Sample Attention Leakage & Multi-Epoch Trade-off

### Symptom: Unexpected validation loss behavior or model outputting mixed content across conversation boundaries when `packing = True`.

#### Cause:
Sequence packing concatenates multiple text items into a single context length window.
1. **Attention Leakage**: If Flash Attention variable-length kernels (`flash_attn_varlen_func`) are inactive, standard SDPA or eager attention applies a single causal mask across the entire concatenated sequence. Token 1 of Sample 2 attends to token 500 of Sample 1!
2. **Multi-Epoch Degredation**: Concatenating samples statically reduces inter-sample randomization and RoPE position embedding diversity across epochs.

#### Fix:
1. Ensure Flash Attention varlen masking is enabled before using `packing = True`.
2. For multi-epoch runs, prefer unpacked SFT training with dynamic sequence batching (`packing = False`).

---

## 7. `get_chat_template()` Overwriting Native Tokenizer Templates

### Symptom: Tokenizer outputs wrong special tokens (e.g. missing `<|im_end|>` or broken `<|eot_id|>`).

#### Cause:
Modern Hugging Face tokenizers (Qwen 3.5, Llama 3.3, Gemma 4) ship with native `tokenizer.chat_template`. Applying `get_chat_template(tokenizer, chat_template="...")` unnecessarily can overwrite official chat templates.

#### Fix:
Use native `tokenizer.apply_chat_template` directly out of the box. Use `get_chat_template()` **only** as a fallback when `tokenizer.chat_template` is missing or when explicitly converting a custom dataset format.

---

## 8. Out-Of-Memory (OOM) Recovery Checklist

If training crashes with `torch.cuda.OutOfMemoryError`:

1. **Set Dropout to Zero**: Ensure `lora_dropout = 0.0` in `get_peft_model` (enables Triton fused path).
2. **Enable Unsloth Checkpointing**: Ensure `use_gradient_checkpointing = "unsloth"` is set in `get_peft_model`.
3. **Use 8-bit Optimizer**: Set `optim = "adamw_8bit"` or `"paged_adamw_8bit"`.
4. **Reduce Batch Size**: Set `per_device_train_batch_size = 1` and scale `gradient_accumulation_steps` up to `8` or `16`.
5. **Enable Tiled MLP**: Pass `unsloth_tiled_mlp = True` in `from_pretrained` for 70B+ parameter models.
6. **Enable Offloading**: Pass `offload_embedding = True` in `from_pretrained`.
