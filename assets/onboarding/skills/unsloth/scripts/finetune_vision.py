#!/usr/bin/env python3
"""
🦥 Unsloth Vision-Language Model (VLM) Fine-Tuning Script

Usage:
    python finetune_vision.py \
        --model_name "unsloth/Qwen2-VL-7B-Instruct" \
        --output_dir "outputs/vlm_model"
"""

import argparse
import sys

import unsloth
from unsloth import FastVisionModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="Unsloth Vision-Language Model Fine-Tuning")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2-VL-7B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="outputs/vlm_model")
    args = parser.parse_args()

    print(f"🦥 Loading Vision-Language Model: {args.model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        text_only=False,  # Keep vision encoder enabled
    )

    print("🦥 Adding LoRA Adapters for Vision-Language model...")
    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("🦥 Configured VLM model for training!")
    print(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Note: Pass image-text conversations dataset to SFTTrainer
    print(f"🦥 VLM fine-tuning setup complete. Save path: {args.output_dir}")


if __name__ == "__main__":
    main()
