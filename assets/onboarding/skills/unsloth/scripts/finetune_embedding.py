#!/usr/bin/env python3
"""
🦥 Unsloth Sentence Transformer / Embedding Model Fine-Tuning Script

Usage:
    uv run python finetune_embedding.py \
        --model_name "BAAI/bge-base-en-v1.5" \
        --output_dir "outputs/embedding_lora"
"""

import argparse
import sys

# ALWAYS import unsloth before transformers/peft/trl
import unsloth
from unsloth import FastSentenceTransformer
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss


def main():
    parser = argparse.ArgumentParser(description="Unsloth Sentence Transformer Fine-Tuning")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="outputs/embedding_lora")
    args = parser.parse_args()

    print(f"🦥 Loading FastSentenceTransformer model: {args.model_name}")
    model = FastSentenceTransformer.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    print("🦥 Adding LoRA Adapters for Embedding Model...")
    model = FastSentenceTransformer.get_peft_model(
        model,
        r=args.r,
        target_modules=["query", "key", "value", "dense"],
        lora_alpha=args.r,
        use_gradient_checkpointing="unsloth",
    )

    print("🦥 Preparing sample contrastive dataset (anchor, positive)...")
    dataset = Dataset.from_dict({
        "anchor": [
            "How to fine-tune LLMs fast?",
            "What is QLoRA?",
            "How to export GGUF models?",
        ] * 20,
        "positive": [
            "Unsloth enables 2-5x faster LLM fine-tuning with 80% lower VRAM.",
            "QLoRA quantizes base weights to 4-bit NormalFloat and adds LoRA adapters.",
            "Use model.save_pretrained_gguf to generate GGUF files for Ollama.",
        ] * 20,
    })

    print("🦥 Initializing MultipleNegativesRankingLoss...")
    loss = MultipleNegativesRankingLoss(model)

    print("🦥 Initializing SentenceTransformerTrainer...")
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=dataset,
        loss=loss,
        args=SentenceTransformerTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            fp16=True,
            logging_steps=1,
        ),
    )

    print("🦥 Starting Sentence Transformer Fine-Tuning...")
    trainer.train()

    print(f"🦥 Saving fine-tuned embedding model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    print("✅ Embedding model fine-tuning complete!")


if __name__ == "__main__":
    main()
