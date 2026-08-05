# Dataset Formatting and Chat Templates Reference Guide

This reference documents chat template formatting, response-only loss masking, ShareGPT standardization, raw text chunking, and synthetic data generation in **Unsloth**.

---

## 1. Chat Templates (`get_chat_template`)

Unsloth standardizes tokenizer chat templates across different model families (Llama 3, Qwen 2.5, DeepSeek, Gemma, ChatML, Zephyr, Alpaca, etc.) and injects EOS token mappings and stop tokens necessary for proper generation termination.

### Entrypoint API
`tokenizer = get_chat_template(tokenizer, chat_template="chatml", ...)`

### Supported Built-In Templates

| Template Identifier | Models / Description | Special Tokens Injected |
| :--- | :--- | :--- |
| `"chatml"` | Standard OpenAI ChatML (`<|im_start|>`, `<|im_end|>`) | `<|im_start|>`, `<|im_end|>` |
| `"qwen-2.5"` | Qwen 2.5 native ChatML variant | `<|im_start|>user\n`, `<|im_start|>assistant\n` |
| `"llama-3"` | Meta Llama 3 / 3.1 / 3.2 / 3.3 Instruct | `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` |
| `"deepseek"` | DeepSeek V2 / V3 / R1 prompt structure | `<|User|>`, `<|Assistant|>`, `<|end_of_sentence|>` |
| `"gemma"` / `"gemma_chatml"` | Google Gemma 1 / 2 / 3 format | `<start_of_turn>user\n`, `<start_of_turn>model\n` |
| `"zephyr"` | HuggingFace Zephyr format | `<|user|>\n`, `<|assistant|>\n` |
| `"unsloth"` | High-efficiency Zephyr-derived template | `>>> User: `, `>>> Assistant: ` |
| `"alpaca"` | Classic Instruction / Input / Output format | `### Instruction:`, `### Response:` |

### Code Example: Applying ChatML Template
```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    map_eos_token = True, # Maps <|im_end|> to tokenizer.eos_token
)
```

---

## 2. Response-Only Loss Masking (`train_on_responses_only`)

By default, standard Causal LM training computes cross-entropy loss over the **entire** token sequence (including system prompts and user questions). This causes models to overfit on prompt style instead of focusing on assistant answers.

Unsloth provides `train_on_responses_only` to automatically set `labels = -100` for all non-assistant tokens.

### Entrypoint API
`trainer = train_on_responses_only(trainer, instruction_part="...", response_part="...")`

### Parameters

| Parameter | Description | Example (ChatML) | Example (Llama 3) |
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

## 3. Dataset Format Standardization (`standardize_sharegpt`)

Unsloth includes data converters in `unsloth.chat_templates` (delegated to `unsloth_zoo.dataset_utils`) to automatically convert arbitrary key schemas (e.g. `from`/`value`, `human`/`gpt`, `queries`/`answers`) into standardized Hugging Face dataset format (`messages`: `[{"role": "user", "content": "..."}, ...]`).

```python
from unsloth.chat_templates import standardize_sharegpt

# Automatically normalizes 'from'/'value', 'conversations', or 'items' into 'messages'
dataset = standardize_sharegpt(dataset)
```

---

## 4. Raw Text & Synthetic Data Loaders

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
# Configured via synthetic_qa_config options
```
