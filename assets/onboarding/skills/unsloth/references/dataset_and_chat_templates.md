# Dataset Formatting and Chat Templates Reference Guide

This reference documents chat template formatting, response-only loss masking, ShareGPT standardization, raw text chunking, and synthetic data generation in **Unsloth**.

---

## 1. Native Chat Templates vs `get_chat_template()`

### Native Hugging Face Chat Templates (Recommended)
Most modern model tokenizers (Qwen 2.5 / 3.5, Llama 3 / 3.1 / 3.3, Gemma 2 / 3 / 4, Mistral, DeepSeek) ship with official, pre-configured chat templates directly inside `tokenizer.chat_template`.

**Agents and developers should prioritize using the native `tokenizer.apply_chat_template(...)` method directly:**

```python
# Modern native chat formatting (No get_chat_template patch required)
text = tokenizer.apply_chat_template(
    conversation,
    tokenize = False,
    add_generation_prompt = False,
)
```

### `get_chat_template()` Fallback Helper
Unsloth's `get_chat_template()` is a legacy helper intended **only** for:
1. Legacy models that lack a `tokenizer.chat_template` defined in their Hugging Face tokenizer config.
2. Overriding/reformatting a dataset with a custom target template (e.g. converting raw instruction data into ChatML or Alpaca format).

```python
from unsloth.chat_templates import get_chat_template

# ONLY use get_chat_template when tokenizer.chat_template is missing or overriding format:
if not getattr(tokenizer, "chat_template", None):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "chatml",
        map_eos_token = True,
    )
```

---

## 2. Supported Template Identifiers (Fallback List)

| Template Identifier | Target Format / Models | Special Tokens Injected |
| :--- | :--- | :--- |
| `"chatml"` | Standard OpenAI ChatML (`<|im_start|>`, `<|im_end|>`) | `<|im_start|>`, `<|im_end|>` |
| `"qwen-2.5"` / `"qwen-3.5"` | Qwen native ChatML format | `<|im_start|>user\n`, `<|im_start|>assistant\n` |
| `"llama-3"` | Meta Llama 3 / 3.1 / 3.2 / 3.3 Instruct | `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` |
| `"deepseek"` | DeepSeek V2 / V3 / R1 prompt structure | `<|User|>`, `<|Assistant|>`, `<|end_of_sentence|>` |
| `"gemma"` | Google Gemma 1 / 2 / 3 / 4 format | `<start_of_turn>user\n`, `<start_of_turn>model\n` |
| `"zephyr"` | HuggingFace Zephyr format | `<|user|>\n`, `<|assistant|>\n` |
| `"alpaca"` | Classic Instruction / Input / Output format | `### Instruction:`, `### Response:` |

---

## 3. Task-Specific Loss Masking Strategies

Loss masking behavior should align with your specific training objective:

- **Chat & Multi-turn Instruction Tuning**: Apply **Assistant-Only Loss** (`train_on_responses_only`) so the model computes gradients solely on assistant answers, preventing overfitting to prompt formatting.
- **Prompt-Completion Datasets**: Apply **Completion-Only Loss** so loss is evaluated strictly after the prompt boundary.
- **Continued Pre-training & Domain Adaptation**: Use **Full-Sequence Loss** (no masking) so the model learns raw language statistics across all tokens.
- **Tool-Calling Datasets**: Apply Assistant Loss while preserving structured `tool_calls` JSON tags and tool outputs in context.

### Entrypoint API (`train_on_responses_only`)
`trainer = train_on_responses_only(trainer, instruction_part="...", response_part="...")`

### Parameters

| Parameter | Description | Example (ChatML / Qwen) | Example (Llama 3) |
| :--- | :--- | :--- | :--- |
| `instruction_part` | Token prefix starting user prompt | `"<|im_start|>user\n"` | `"<|start_header_id|>user<|end_header_id|>\n\n"` |
| `response_part` | Token prefix starting assistant prompt | `"<|im_start|>assistant\n"` | `"<|start_header_id|>assistant<|end_header_id|>\n\n"` |

### Code Example: Response-Only Masking Setup
```python
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(...)

# Mask everything except assistant responses
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

trainer.train()
```

---

## 4. Dataset Format Standardization (`standardize_sharegpt`)

Unsloth includes data converters in `unsloth.chat_templates` (delegated to `unsloth_zoo.dataset_utils`) to automatically convert arbitrary key schemas (e.g. `from`/`value`, `human`/`gpt`, `queries`/`answers`) into standardized Hugging Face dataset format (`messages`: `[{"role": "user", "content": "..."}, ...]`).

```python
from unsloth.chat_templates import standardize_sharegpt

# Automatically normalizes 'from'/'value', 'conversations', or 'items' into 'messages'
dataset = standardize_sharegpt(dataset)
```

---

## 5. Raw Text & Synthetic Data Loaders

### `RawTextDataLoader`
Loads unstructured text files (`.txt`, `.md`, `.json`, `.jsonl`, `.csv`) and chunks them into overlapping causal LM sequences:

```python
from unsloth import RawTextDataLoader

loader = RawTextDataLoader(
    tokenizer = tokenizer,
    chunk_size = 2048,
    stride = 512,
)

dataset = loader.load_from_file("document.txt")
```

### `SyntheticDataKit`
Uses local vLLM or fast sampling models to generate synthetic Q&A data or dataset expansion.

```python
from unsloth import SyntheticDataKit

kit = SyntheticDataKit()
```
