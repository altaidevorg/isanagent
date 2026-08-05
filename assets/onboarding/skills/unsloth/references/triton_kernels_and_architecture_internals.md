# Triton Fused Kernels and Architecture Internals Reference Guide

This reference provides a deep-dive technical breakdown of **Unsloth's custom Triton kernels**, activation memory elimination, Mixture of Experts (MoE) routing, and internal architecture patches.

---

## 1. Why Unsloth is 2-5x Faster with 80% Less VRAM

Standard PyTorch implementations of LoRA and Transformer layers store massive intermediate activation tensors during the forward pass to compute gradients during autograd backward passes. For large context lengths or high ranks, these activations cause GPU Out-Of-Memory (OOM) errors.

Unsloth replaces standard PyTorch linear layers, activation functions, and normalization modules with **hand-written, fused Triton C-like GPU kernels**:

```
[PyTorch Standard]
Input -> Matrix Mult -> Store Activation -> LoRA A -> Store Activation -> LoRA B -> Output
          (High VRAM)                       (High VRAM)

[Unsloth Fused Triton]
Input -> [ Fused Kernel: Base MatMul + LoRA A + LoRA B + Activation ] -> Output
          (Zero intermediate activation storage - Recomputed on backward pass)
```

---

## 2. Key Triton Fused Kernels in Unsloth

| Kernel Module | Source File | Mathematical Operation / Optimization |
| :--- | :--- | :--- |
| **Fast LoRA Forward & Backward** | `kernels/fast_lora.py` | Computes $(W + \frac{\alpha}{r} A \cdot B) X$ in a single fused GPU block without allocating intermediate LoRA output tensors. |
| **Fast Cross Entropy Loss** | `kernels/cross_entropy_loss.py` | Fuses Softmax and Cross-Entropy loss computation directly over vocabulary logits, avoiding $O(\text{batch} \times \text{seq} \times \text{vocab})$ memory allocation. |
| **Fast RoPE Embeddings** | `kernels/rope_embedding.py` | Applies Rotary Position Embeddings in-place directly on Attention Q and K matrices without copying tensors. |
| **Fused RMSNorm & LayerNorm** | `kernels/rms_layernorm.py` | Computes Root Mean Square normalization in a single warp reduction step. |
| **Fused SwiGLU & GeGLU** | `kernels/swiglu.py` / `geglu.py` | Fuses elementwise gate multiplications $\text{Swish}(X_{\text{gate}}) \cdot X_{\text{up}}$ in GPU SRAM. |
| **FP8 Blockwise Quantization** | `kernels/fp8.py` | Block-level 8-bit floating point (E4M3 / E5M2) scaling and dequantization. |

---

## 3. Mixture of Experts (MoE) Routing & Fine-Tuning

Unsloth optimizes MoE architectures (such as **Qwen3-MoE**, **GLM-4 MoE**, and **DeepSeek V3/R1 MoE**):

- **Targeting MoE Experts**: When calling `get_peft_model`, target both top-level attention projections and router/expert gate matrices (`gate_proj`, `up_proj`, `down_proj` or `experts`).
- **Expert Parameter Filtering**: Unsloth handles gating parameters (`gate`, `router`) separately to avoid rank degradation across sparse routing layers.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "gate", "shared_expert", "experts", # Targets MoE sparse router & expert layers
    ],
    lora_alpha = 16,
    use_gradient_checkpointing = "unsloth",
)
```

---

## 4. Attention Dispatch Hierarchy (`attention_dispatch.py`)

Unsloth checks GPU compute capabilities dynamically and dispatches to the optimal attention kernel:

1. **FlashAttention-2** (`flash_attn`): Selected for NVIDIA Ampere/Hopper/Ada GPUs.
2. **FlexAttention** (`flex_attention`): Selected when custom masking or PyTorch 2.5+ flex kernels are enabled.
3. **Flash-Linear-Attention** (`fla`): Selected for Gated-DeltaNet / Linear-Attention hybrid models (Qwen 3.5).
4. **SDPA (Scaled Dot Product Attention)**: Fallback PyTorch standard kernel.
