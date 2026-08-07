#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth>=2026.8.0",
#     "unsloth_zoo>=2026.8.0",
#     "torch>=2.13.0",
#     "transformers>=5.14.1",
#     "peft>=0.20.0",
#     "trl>=1.9.2",
#     "datasets>=5.0.1",
#     "accelerate>=1.14.0",
#     "pillow>=12.3.0",
# ]
# ///
"""
🦥 Unsloth Vision-Language Model (VLM) Fine-Tuning Script

Usage:
    python finetune_vision.py \
        --model_name "unsloth/Qwen2.5-VL-7B-Instruct" \
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
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True, help="Enable 4-bit NF4 QLoRA quantization")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="outputs/vlm_model")
    args = parser.parse_args()

    print(f"🦥 Loading Vision-Language Model: {args.model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        text_only=False,  # Keep vision encoder enabled
    )

    print("🦥 Adding LoRA Adapters for Vision-Language model...")
    model = FastVisionModel.get_peft_model(
        model,
        r=args.r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("🦥 Loading sample VLM dataset...")
    # Load sample dataset with image-text content
    dataset = load_dataset("unsloth/radiology_mini", split="train[:100]")

    def format_vlm_prompts(examples):
        texts = []
        for caption in examples["caption"]:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": "Describe this image in detail."}]},
                {"role": "assistant", "content": [{"type": "text", "text": caption}]},
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    dataset = dataset.map(format_vlm_prompts, batched=True)

    print("🦥 Initializing SFTTrainer for VLM Fine-Tuning...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=None,  # Note: TRL recommends max_length=None for VLMs to avoid truncating image tokens
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            seed=3407,
            output_dir=args.output_dir,
        ),
    )

    print("🦥 Starting Vision-Language Model Fine-Tuning...")
    trainer.train()

    print(f"🦥 Saving fine-tuned VLM model adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ VLM Fine-Tuning complete!")


if __name__ == "__main__":
    main()
