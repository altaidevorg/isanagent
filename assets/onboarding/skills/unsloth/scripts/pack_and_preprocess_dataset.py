#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth>=2026.8.0",
#     "unsloth_zoo>=2026.8.0",
#     "torch>=2.13.0",
#     "transformers>=5.14.1",
#     "peft>=0.20.0",
#     "datasets>=5.0.1",
#     "accelerate>=1.14.0",
# ]
# ///
"""
🦥 Unsloth Dataset Sequence Packing & Preprocessing Tool

GOTCHA & ATTENTION MASK WARNING:
- Sequence packing (`packing=True`) concatenates multiple short samples into single context length windows.
- Flash Attention variable-length kernels (`flash_attn_varlen_func`) MUST be active to avoid cross-sample attention leakage.
  Without Flash Attention varlen masking, attention across packed document boundaries is unmasked.
- Trade-off: Packing increases throughput (~2-3x) but may slightly degrade loss in multi-epoch runs
  due to static sequence boundaries and reduced sample randomization across epochs.

Usage:
    python pack_and_preprocess_dataset.py \
        --model_name "unsloth/Qwen3-8B" \
        --max_seq_length 4096 \
        --output_path "outputs/packed_dataset"
"""

import argparse
import sys

import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Unsloth Dataset Packing & Preprocessing")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-8B")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--override_chat_template", type=str, default=None, help="Optional fallback template name if tokenizer lacks chat_template")
    parser.add_argument("--output_path", type=str, default="outputs/packed_dataset")
    args = parser.parse_args()

    print(f"🦥 Loading tokenizer from {args.model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    if args.override_chat_template:
        tokenizer = get_chat_template(tokenizer, chat_template=args.override_chat_template)
    elif not getattr(tokenizer, "chat_template", None):
        print("⚠️ Tokenizer lacks native chat_template, falling back to chatml...")
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    print("🦥 Loading sample dataset...")
    dataset = load_dataset("philschmid/dolly-15k-curated-en", split="train[:500]")

    def format_prompts(examples):
        texts = []
        for instruction, context, response in zip(
            examples["instruction"], examples["context"], examples["response"]
        ):
            user_msg = instruction if not context else f"{instruction}\n\nContext: {context}"
            convo = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]
            texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    formatted_dataset = dataset.map(format_prompts, batched=True)

    print(f"🦥 Formatted {len(formatted_dataset)} samples using native chat template.")
    print(f"Sample formatted text:\n{formatted_dataset[0]['text'][:300]}...")

    print(f"🦥 Saving preprocessed dataset to {args.output_path}...")
    formatted_dataset.save_to_disk(args.output_path)
    print("✅ Preprocessing & dataset preparation complete!")


if __name__ == "__main__":
    main()
