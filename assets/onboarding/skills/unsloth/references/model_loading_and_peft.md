# Model Loading and PEFT (LoRA / QLoRA) Reference Guide

This reference provides exhaustive technical documentation for loading models, configuring quantization, applying PEFT/LoRA adapters, adjusting RoPE scaling, and tuning attention dispatch kernels in **Unsloth**.

---

## 1. Model Loading (`FastLanguageModel` & `FastVisionModel`)

Unsloth provides optimized model wrappers that patch Hugging Face Transformers and PyTorch internals upon loading to accelerate model loading, lower memory footprint, and optimize autograd backward passes.

### Key Entrypoint APIs
- `FastLanguageModel.from_pretrained(...)`: For Causal Language Models (Llama 3, Qwen 2.5, DeepSeek, Gemma 2/3, Mistral, Granite, Cohere, etc.).
- `FastVisionModel.from_pretrained(...)`: For Vision-Language Models (Qwen2-VL, Llama-3.2-Vision, Pixtral, etc.).
- `FastModel.from_pretrained(...)`: Base polymorphic entrypoint.

### Parameters of `from_pretrained`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `str` | `"unsloth/Llama-3.2-1B-Instruct"` | Hugging Face repo ID or local checkpoint directory. Pre-quantized 4-bit repos (e.g., `unsloth/llama-3-8b-bnb-4bit`) load much faster. |
| `max_seq_length` | `int` | `2048` | Maximum context sequence length. Automatically extends RoPE scaling when loading sequences larger than model native context. |
| `dtype` | `torch.dtype` / `None` | `None` | Precision type (`torch.bfloat16`, `torch.float16`, `torch.float32`). If `None`, auto-detected based on GPU compute capability (bfloat16 on Ampere/Hopper/Ada, float16 on Turing/Pascal). |
| `load_in_4bit` | `bool` | `True` | Enables 4-bit NormalFloat (NF4) quantization via `bitsandbytes` (QLoRA). Reduces VRAM usage by ~70-80%. |
| `load_in_8bit` | `bool` | `False` | Enables 8-bit quantization. |
| `load_in_16bit` | `bool` | `False` | Loads model in native 16-bit precision without weight quantization. |
| `load_in_fp8` | `bool` | `False` | Enables FP8 quantization (E4M3 / E5M2) via PyTorch / torchao for supported architectures. |
| `full_finetuning` | `bool` | `False` | Set `True` to enable full parameter fine-tuning (requires `UNSLOTH_ENABLE_FULL_FINETUNING=1`). |
| `device_map` | `str` / `dict` | `"sequential"` | Device allocation map. On multi-GPU distributed runs (`torchrun`), Unsloth automatically assigns local GPU ranks to prevent Accelerate device mismatch errors. |
| `rope_scaling` | `dict` / `None` | `None` | Optional RoPE scaling configuration (e.g. `{"type": "dynamic", "factor": 2.0}` or YaRN/linear scaling). |
| `fast_inference` | `bool` | `False` | Boots vLLM engine alongside model loading for high-speed generation loops. |
| `gpu_memory_utilization` | `float` | `0.5` | Fractional VRAM allocated to vLLM engine when `fast_inference=True`. |
| `text_only` | `bool` | `False` | For VLMs: if `True`, strips vision tower and loads text decoder only. |

### Code Example: Loading a Model with 4-bit QLoRA
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length = 8192,
    dtype = None,           # Auto-select bf16 if supported, else fp16
    load_in_4bit = True,    # 4-bit NF4 QLoRA
    trust_remote_code = False,
)
```

---

## 2. Parameter-Efficient Fine-Tuning (`get_peft_model`)

Unsloth replaces default PEFT/LoRA modules with custom Triton fused kernels for forward and backward passes. This removes intermediate activation caching for LoRA matrix multiplications.

### Key Entrypoint API
`FastLanguageModel.get_peft_model(model, ...)`

### Parameters of `get_peft_model`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `PreTrainedModel` | *Required* | Model object returned from `from_pretrained`. |
| `r` | `int` | `16` | LoRA rank dimension. Recommended values: 8, 16, 32, 64, 128. |
| `target_modules` | `list[str]` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Modules to attach LoRA adapters to. Target all 7 linear layers for maximum capacity. |
| `lora_alpha` | `int` | `16` | LoRA scaling factor. Usually set equal to `r` or `2 * r`. |
| `lora_dropout` | `float` | `0.0` | Dropout probability. **Set to `0.0`** for optimal speed (enables Triton fused path). |
| `bias` | `str` | `"none"` | Bias fine-tuning strategy (`"none"`, `"all"`, `"lora_only"`). Always use `"none"` unless necessary. |
| `use_gradient_checkpointing` | `str` / `bool` | `"unsloth"` | Gradient checkpointing method (`"unsloth"`, `"unsloth_offload"`, `True`, `False`). `"unsloth"` recomputes activations using fused kernels, saving 80% VRAM. |
| `use_rslora` | `bool` | `False` | Enables Rank-Stabilized LoRA (`lora_alpha / sqrt(r)` scaling). Prevents instability at high ranks. |
| `use_dora` | `bool` | `False` | Enables Weight-Decomposed Low-Rank Adaptation (DoRA). |
| `modules_to_save` | `list[str]` | `None` | Non-LoRA modules to make trainable (e.g. `["embed_tokens", "lm_head"]` for token vocabulary extension). |
| `random_state` | `int` | `3407` | Random seed for initialization. |

### Code Example: Standard 7-Module LoRA Setup
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = 16,
    lora_dropout = 0.0,  # 0.0 activates optimized fused kernels
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

---

## 3. Rotary Position Embeddings (RoPE) & Context Length

Unsloth automatically manages RoPE scaling when sequence lengths exceed default pre-training boundaries.

- **Automatic inv_freq recalculation**: For Llama 3 / 3.1 / 3.2 / 3.3 models, Unsloth patches rotary embedding classes to prevent position code corruption.
- **YaRN and Dynamic Scaling**: Can be passed directly via `rope_scaling` in `from_pretrained`:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    max_seq_length = 32768,
    rope_scaling = {"type": "dynamic", "factor": 4.0},
)
```

---

## 4. Attention Backends & Fused Kernels

Unsloth automatically dispatches to the fastest available attention implementation:

1. **FlashAttention-2**: Enabled automatically on Ampere (A100, RTX 3090/4090), Hopper (H100), and Ada GPUs.
2. **FlexAttention**: PyTorch 2.5+ flexible attention kernel support.
3. **SDPA (Scaled Dot-Product Attention)**: Native PyTorch fallback when FlashAttention is absent.
4. **FLA (Flash-Linear-Attention)**: Fast Triton kernels for Gated-DeltaNet linear attention models (e.g. Qwen 3.5, Kimi-Linear).
