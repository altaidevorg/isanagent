#!/usr/bin/env python3
"""
🦥 Unsloth Dataset Sequence Packing & Preprocessing Tool

Usage:
    uv run python pack_and_preprocess_dataset.py \
        --model_name "unsloth/Qwen2.5-7B-Instruct" \
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
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--chat_template", type=str, default="qwen-2.5")
    parser.add_argument("--output_path", type=str, default="outputs/packed_dataset")
    args = parser.parse_args()

    print(f"🦥 Loading tokenizer from {args.model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

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

    print(f"🦥 Formatted {len(formatted_dataset)} samples into chat template format.")
    print(f"Sample formatted text:\n{formatted_dataset[0]['text'][:300]}...")

    print(f"🦥 Saving preprocessed dataset to {args.output_path}...")
    formatted_dataset.save_to_disk(args.output_path)
    print("✅ Preprocessing & dataset preparation complete!")


if __name__ == "__main__":
    main()
