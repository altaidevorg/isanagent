# Sequence Packing and Prefix Grouping Reference Guide

This reference documents **Sequence Packing** (`packing.py`) and **Prefix Grouping / Prompt Caching** (`prefix_grouper.py`) in **Unsloth**.

---

## 1. Sequence Packing (0% Padding Waste)

During standard dataset training, samples are padded with `<pad>` tokens to match the longest sequence in a batch. For datasets with variable lengths (e.g. short answers mixed with long explanations), padding tokens can consume **50% to 80%** of total compute.

### How Sequence Packing Works

Unsloth concatenates multiple short text samples into a single continuous sequence of exact length `max_seq_length`, separated by `<eos>` tokens, with **zero padding tokens**:

```
[Standard Batching with Padding (Wasteful)]
Batch Item 1: [ User Q1 ... Assistant A1 ... <pad> <pad> <pad> <pad> ]  (50% Waste)
Batch Item 2: [ User Q2 ... Assistant A2 ... <pad> <pad> <pad> ]        (40% Waste)

[Unsloth Packed Batching (100% Efficiency)]
Packed Item:  [ User Q1 .. A1 <eos> User Q2 .. A2 <eos> User Q3 .. A3 <eos> ]  (0% Waste)
```

### Enabling Packing in `SFTTrainer`

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 4096,
    packing = True, # Concatenates short samples to fill max_seq_length
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        output_dir = "outputs/packed_sft",
    ),
)
```

---

## 2. Prefix Grouping and Prompt Caching in RL (`prefix_grouper.py`)

In Reinforcement Learning algorithms like **GRPO** (Group Relative Policy Optimization), the model generates $G$ (e.g. 8 or 16) completion candidates for **the exact same prompt**.

### Prefix Grouper Kernel
Unsloth's `PrefixGrouper` kernel identifies identical prompt prefixes across generation groups:

1. **Shared KV Cache**: Computes prompt KV cache **once** per group instead of re-computing $G$ times.
2. **Memory Savings**: Reduces prompt encoding memory overhead by $G \times$.
3. **vLLM Integration**: Automatically used when `use_vllm = True` in `GRPOConfig`.

```python
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig

# Patch Fast RL initializes PrefixGrouper & Triton Cross-Entropy automatically
PatchFastRL("GRPO", FastLanguageModel)
```
