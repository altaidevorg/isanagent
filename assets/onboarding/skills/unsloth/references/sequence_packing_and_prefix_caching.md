# Sequence Packing and Prefix Grouping Reference Guide

This reference documents **Sequence Packing** (`packing.py`) and **Prefix Grouping / Prompt Caching** (`prefix_grouper.py`) in **Unsloth**, including critical attention masking requirements and trade-offs.

---

## 1. Sequence Packing & Attention Mask Gotchas

During standard dataset training, samples are padded with `<pad>` tokens to match the longest sequence in a batch. Sequence packing concatenates multiple short text samples into a single continuous sequence of exact length `max_seq_length`.

```
[Standard Batching with Padding]
Batch Item 1: [ User Q1 ... Assistant A1 ... <pad> <pad> <pad> <pad> ]
Batch Item 2: [ User Q2 ... Assistant A2 ... <pad> <pad> <pad> ]

[Sequence Packing]
Packed Item:  [ User Q1 .. A1 <eos> User Q2 .. A2 <eos> User Q3 .. A3 <eos> ]
```

### Critical Technical Gotchas & Requirements

1. **Flash Attention Varlen Requirement (`flash_attn_varlen_func`)**:
   - Without Flash Attention variable-length kernels (`cu_seqlens`), standard PyTorch SDPA, eager attention, or naive attention kernels apply a **single causal triangular mask across the entire concatenated sequence**.
   - This causes **cross-document attention leakage**: token 1 of Sample 2 attends to token 500 of Sample 1!
   - **Requirement**: Always verify Flash Attention variable-length masking (`flash_attn`) is active when enabling `packing = True`.

2. **Multi-Epoch Randomization Trade-off**:
   - **Throughput vs Performance**: Sequence packing increases throughput (~2x–3x) by removing padding.
   - **Randomization Loss**: Pre-packed datasets or static packing algorithms keep the exact same sequence pairs joined together across all epochs.
   - **Impact**: In multi-epoch training runs, static sequence pairing reduces position embedding (RoPE) diversity and inter-sample randomization, which can cause slight performance degradation or higher evaluation loss compared to unpacked batches with dynamic shuffling.

### Enabling Packing in `SFTTrainer`

```python
from trl import SFTTrainer, SFTConfig

# Use packing ONLY when Flash Attention varlen kernels are available
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = 4096,
        packing = True, # Set True for fast text packing (Flash Attention required)
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
