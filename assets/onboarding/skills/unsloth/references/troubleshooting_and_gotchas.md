# Unsloth Comprehensive Troubleshooting, Gotchas, and Best Practices Guide

This guide documents common operational failure modes, NaN loss diagnosis, precision rules, CUDA linker fixes, vLLM memory conflicts, Tesla T4 FlashInfer gotchas, DDP multi-GPU traps, 16GB VRAM optimization recipes, and debugging strategies in **Unsloth**.

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
    model_name = "unsloth/Qwen2.5-7B-Instruct",
    device_map = "sequential", # Unsloth automatically remaps to local rank in DDP
    load_in_4bit = True,
)
```

---

## 5. Gradient Checkpointing & Input Require Grads Error

### Symptom: `RuntimeError: Element 0 of tensors does not require grad and does not have a grad_fn`.

#### Cause:
Quantized 4-bit base model parameters frozen without enabling input gradients before attaching LoRA adapters.

#### Fix:
Unsloth handles this automatically inside `get_peft_model`. If writing custom training loops outside `SFTTrainer`, call:

```python
model.enable_input_require_grads()
```

---

## 6. vLLM Engine VRAM Allocation Conflicts in GRPO / Inference

### Symptom: CUDA Out-Of-Memory (OOM) during vLLM initialization (`fast_inference=True` or `GRPOTrainer(use_vllm=True)`).

#### Cause:
By default, vLLM pre-allocates 90% of free GPU VRAM for KV cache blocks, leaving insufficient VRAM for PyTorch model gradients and optimizer states.

#### Fix:
Limit vLLM GPU memory utilization to 40-50% when co-locating training and generation on the same GPU:

```python
# For FastLanguageModel loading:
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Math-7B-Instruct",
    fast_inference = True,
    gpu_memory_utilization = 0.4, # Reserve VRAM for training gradients
)

# For GRPOConfig:
args = GRPOConfig(
    use_vllm = True,
    vllm_gpu_memory_utilization = 0.4,
)
```

---

## 7. Tesla T4 GPU & FlashInfer Compatibility (GRPO / vLLM)

### Symptom: vLLM initialization fails or crashes with FlashInfer kernel errors on Tesla T4 GPUs (Colab / Kaggle).

#### Cause:
Tesla T4 GPUs (compute capability 7.5) do not support FlashInfer's optimized top-p/top-k sampling kernels, which vLLM attempts to load by default.

#### Fix:
Disable FlashInfer by setting `UNSLOTH_VLLM_NO_FLASHINFER="1"` before importing `unsloth`:

```python
import os
os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"

import unsloth
from unsloth import FastLanguageModel
```

---

## 8. CUDA Runtime Shared Library Path Resolution (Colab VM)

### Symptom: PyTorch / vLLM fails to locate `libcudart.so` or `nvidia/cuXX/lib` runtime libraries on Colab or Linux VMs.

#### Cause:
CUDA shared libraries installed inside Python site-packages are missing from system `LD_LIBRARY_PATH`.

#### Fix:
Append `nvidia/cuXX/lib` to `LD_LIBRARY_PATH` at the top of your Python script:

```python
import os
import sys
# Automatically locate nvidia CUDA lib directory inside python site-packages
for p in sys.path:
    nvidia_path = os.path.join(p, "nvidia")
    if os.path.isdir(nvidia_path):
        for sub in os.listdir(nvidia_path):
            lib_dir = os.path.join(nvidia_path, sub, "lib")
            if os.path.isdir(lib_dir):
                os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + lib_dir
```

---

## 9. `colab-cli` AttributeError with `jupyter_kernel_client`

### Symptom: `colab-cli` fails with `AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'`.

#### Cause:
`jupyter_kernel_client` 1.0.0+ renamed `KernelClient` to `JupyterKernelClient`.

#### Fix:
Downgrade `jupyter_kernel_client` or alias `KernelClient = JupyterKernelClient` before importing `colab_cli`.

---

## 10. 16GB VRAM (Tesla T4 / RTX 4060Ti) Guarantee Recipe

To guarantee zero OOM errors when fine-tuning 7B models on 16GB VRAM GPUs:

1. **Gradient Accumulation**: Set `per_device_train_batch_size = 1` and increase `gradient_accumulation_steps = 8` or `16`.
2. **Zero Dropout (`lora_dropout = 0.0`)**: Crucial because Unsloth's Triton custom fused kernels (which eliminate intermediate activation caching and save ~80% VRAM) require `lora_dropout = 0.0`.
3. **Sequence Length**: Keep `max_seq_length = 1024` or `2048`.
4. **Implicit Reference Model in DPO**: Always set `ref_model = None` in `DPOTrainer`. Unsloth computes reference logits on-the-fly, saving ~5.5 GB of VRAM.

---

## 11. Broken Compiled Extensions (`causal_conv1d`, `vllm`, `fbgemm_gpu`)

### Symptom: `ImportError: undefined symbol` or C++ extension segfaults on startup.

#### Cause:
Mismatched PyTorch, CUDA, or C++ ABI versions in pre-built binary wheels.

#### Fix:
Unsloth detects and disables broken C++ extensions automatically (`disable_broken_vllm()`). Re-install clean builds with `uv`:

```bash
uv pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

---

## 12. RNG Checkpoint Resume Crashes in TRL

### Symptom: `RuntimeError: Expected all tensors to be on the same device` or unpickling failure when resuming training from a checkpoint (`resume_from_checkpoint=True`).

#### Cause:
TRL's default RNG state saver attempts to reload CUDA RNG states for quantized model buffers across mismatched device IDs.

#### Fix:
Unsloth applies `patch_unsafe_trainer_rng_load()` automatically. If using custom Trainer callbacks, set `ignore_data_skip=True` in `SFTConfig`.

---

## 13. Apple Silicon MLX Metal Context Timeout & MPS Errors

### Symptom: Metal command buffer timeout or crash on M1/M2/M3/M4 Macs.

#### Cause:
AGX Metal CDM context store timeout during long generation loops.

#### Fix:
Unsloth sets `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` automatically at import time. On macOS, ensure you do not pass `device_map="cuda"` or `.to("cuda")` in user code.

---

## 14. WandB / TensorBoard Logging Deadlocks in Multi-GPU Jobs

### Symptom: Training process hangs indefinitely at step 0 during logger initialization under `torchrun`.

#### Cause:
WandB service child process deadlocking when spawned under PyTorch DDP multiprocessing.

#### Fix:
Use `report_to="tensorboard"` or set `WANDB_START_METHOD="thread"` in your shell environment:

```bash
export WANDB_START_METHOD="thread"
```

---

## 15. GGUF Export Missing Tokenizer / Special Tokens

### Symptom: `llama.cpp` error: `invalid token id` or missing chat template in exported GGUF file.

#### Cause:
Calling `save_pretrained_gguf` on a model without passing the associated `tokenizer` object.

#### Fix:
Always pass `tokenizer` as the second argument:

```python
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")
```

---

## 16. Out-Of-Memory (OOM) Recovery Checklist

If training crashes with `torch.cuda.OutOfMemoryError`:

1. **Set Dropout to Zero**: Ensure `lora_dropout = 0.0` in `get_peft_model` (enables Triton fused path).
2. **Enable Unsloth Checkpointing**: Ensure `use_gradient_checkpointing = "unsloth"` is set in `get_peft_model`.
3. **Use 8-bit Optimizer**: Set `optim = "adamw_8bit"` or `"paged_adamw_8bit"`.
4. **Reduce Batch Size**: Set `per_device_train_batch_size = 1` and scale `gradient_accumulation_steps` up to `8` or `16`.
5. **Enable Tiled MLP**: Pass `unsloth_tiled_mlp = True` in `from_pretrained` for 70B+ parameter models.
6. **Enable Offloading**: Pass `offload_embedding = True` in `from_pretrained`.
