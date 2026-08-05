# Advanced Topics, Architectures, and Optimizers Reference Guide

This reference provides technical documentation for specialized model architectures, advanced optimizers, Tiled MLP, Quantized Activation Training (QAT), and Diffusion models in **Unsloth**.

---

## 1. Sentence Transformers & Embedding Fine-Tuning (`FastSentenceTransformer`)

Unsloth provides high-performance fine-tuning for embedding and retrieval models (e.g. BGE, E5, MiniLM, RoBERTa, EmbeddingGemma).

### Entrypoint API
`FastSentenceTransformer.from_pretrained(...)`

```python
from unsloth import FastSentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss

# 1. Load Sentence Transformer in 4-bit / 16-bit
model = FastSentenceTransformer.from_pretrained(
    model_name = "BAAI/bge-base-en-v1.5",
    max_seq_length = 512,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters
model = FastSentenceTransformer.get_peft_model(
    model,
    r = 16,
    target_modules = ["query", "key", "value", "dense"],
    lora_alpha = 16,
    use_gradient_checkpointing = "unsloth",
)

# 3. Configure MultipleNegativesRankingLoss & Trainer
loss = MultipleNegativesRankingLoss(model)

trainer = SentenceTransformerTrainer(
    model = model,
    train_dataset = dataset,
    loss = loss,
    args = SentenceTransformerTrainingArguments(
        output_dir = "outputs/bge_lora",
        per_device_train_batch_size = 32,
        learning_rate = 2e-4,
        max_steps = 100,
        fp16 = True,
    ),
)
trainer.train()
```

---

## 2. Advanced Memory Optimizations

### Tiled MLP (`unsloth_tiled_mlp=True`)
For models with very large intermediate MLP dimensions (e.g. Llama 3 70B, Qwen 2.5 72B), intermediate activation memory during SwiGLU / GeGLU forward passes can dominate VRAM.

Unsloth tiles MLP projections across chunked sequence dimensions:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
    max_seq_length = 8192,
    load_in_4bit = True,
    unsloth_tiled_mlp = True, # Tiles MLP activations to save 30-40% peak VRAM
)
```

### Q-GaLore & Projection Optimizers
Q-GaLore performs low-rank gradient projections combined with quantized optimizer states, allowing full parameter or large-rank tuning with minimal memory overhead.

```python
# Select optimizer in TrainingArguments:
optim = "q_galore_adamw"  # Custom low-rank gradient projection optimizer
```

### Vocabulary Extension (`modules_to_save`)
When adding new domain-specific tokens to a model tokenizer, extend the embedding matrix and output head parameters:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save = ["embed_tokens", "lm_head"], # Trains embedding & output head
)
```

---

## 3. Quantization-Aware Training (QAT)

QAT simulates 4-bit / 8-bit quantization noise during training forward passes while maintaining float weights, resulting in higher quality quantized checkpoints post-training.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct",
    load_in_4bit = False,
    qat_scheme = "int4", # Enables QAT simulation
)
```

---

## 4. Diffusion Models Fine-Tuning (`FastDiffusionModel`)

Unsloth extends LoRA optimization and Triton kernels to text-to-image and image-generation diffusion models (e.g. FLUX, Stable Diffusion).

```python
from unsloth.models.diffusion import FastDiffusionModel

model = FastDiffusionModel.from_pretrained(
    model_name = "black-forest-labs/FLUX.1-schnell",
    load_in_4bit = True,
)

model = FastDiffusionModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"],
)
```
