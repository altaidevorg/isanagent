# Saving, Export, and Deployment Reference Guide

This reference provides complete instructions for saving fine-tuned models, merging LoRA adapters, converting checkpoints to GGUF format, generating Ollama Modelfiles, uploading to Hugging Face Hub, and serving with vLLM in **Unsloth**.

---

## 1. Saving LoRA Adapters & Merging Weights

Unsloth monkey-patches saving functions onto model and tokenizer objects via `unsloth.save.patch_saving_functions`.

### Saving Options Overview

| Save Method | Method Name | Description |
| :--- | :--- | :--- |
| **LoRA Adapters Only** | `model.save_pretrained(dir)` | Saves only lightweight adapter weights (`adapter_model.safetensors` + `adapter_config.json`, ~50-200MB). |
| **Merged 16-bit** | `model.save_pretrained_merged(dir, tokenizer, save_method="merged_16bit")` | Dequantizes base model, merges LoRA adapters, and saves full 16-bit float16/bfloat16 weights. |
| **Merged 4-bit** | `model.save_pretrained_merged(dir, tokenizer, save_method="merged_4bit")` | Merges LoRA adapters directly into 4-bit quantized base model weights. |
| **GGUF Quantized** | `model.save_pretrained_gguf(dir, tokenizer, quantization_method=...)` | Converts merged model to GGUF format for llama.cpp / Ollama / LM Studio. |

### Code Example: Merging & Local Saving
```python
# 1. Save LoRA Adapters only
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# 2. Save Merged 16-bit Full Model
model.save_pretrained_merged("merged_16bit_model", tokenizer, save_method = "merged_16bit")

# 3. Save Merged 4-bit Model
model.save_pretrained_merged("merged_4bit_model", tokenizer, save_method = "merged_4bit")
```

---

## 2. GGUF Conversion & Quantization (`save_pretrained_gguf`)

Unsloth bundles high-speed `llama.cpp` quantization pipelines directly into Python.

### Supported GGUF Quantization Types

| Quant Method | Quality / Size Trade-off | Recommended Use Case |
| :--- | :--- | :--- |
| `"q4_k_m"` | **Recommended**. Balanced precision and medium size. | Best default for local LLM execution. |
| `"q8_0"` | High quality, large file size. | Near-16bit accuracy for desktop GPUs. |
| `"f16"` / `"bf16"` | Unquantized Float16 / Bfloat16 GGUF. | Lossless conversion for server deployment. |
| `"q5_k_m"` | High quality, slightly larger than Q4. | Excellent reasoning model balance. |
| `"q2_k"` / `"q3_k_m"` | Ultra-compressed 2-bit / 3-bit. | Low-memory edge devices / phones. |

### Code Example: Converting to GGUF
```python
# Convert and save Q4_K_M GGUF locally
model.save_pretrained_gguf(
    "model_gguf_output",
    tokenizer,
    quantization_method = "q4_k_m",
)

# Convert and save multiple quantization formats
model.save_pretrained_gguf(
    "model_gguf_multi",
    tokenizer,
    quantization_method = ["q4_k_m", "q8_0", "f16"],
)
```

---

## 3. Hugging Face Hub & Ollama Direct Export

Unsloth can convert models to GGUF, generate a matched Ollama `Modelfile`, and push them directly to Hugging Face Hub in a single command.

```python
# Push Merged 16-bit Model to HF Hub
model.push_to_hub_merged(
    "username/my-finetuned-llama3",
    tokenizer,
    save_method = "merged_16bit",
    token = "hf_...",
)

# Push GGUF + Ollama Modelfile directly to HF Hub
model.push_to_hub_gguf(
    "username/my-finetuned-llama3-gguf",
    tokenizer,
    quantization_method = ["q4_k_m", "q8_0"],
    token = "hf_...",
)
```

### Running with Ollama Locally
Once pushed to Hugging Face, run directly in terminal via Ollama:
```bash
ollama run hf.co/username/my-finetuned-llama3-gguf:Q4_K_M
```

---

## 4. High-Speed Inference with vLLM (`for_inference`)

To use fine-tuned Unsloth models for high-throughput inference, enable vLLM decoding mode:

```python
from unsloth import FastLanguageModel

# Enable Fast Inference mode
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize = False,
            add_generation_prompt = True,
        )
    ],
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print(tokenizer.batch_decode(outputs))
```
