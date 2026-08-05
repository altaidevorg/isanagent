#!/usr/bin/env python3
"""
🦥 Unsloth Direct Preference Optimization (DPO) Fine-Tuning Script

Usage:
    uv run python finetune_dpo.py \
        --model_name "unsloth/Qwen2.5-7B-Instruct" \
        --output_dir "outputs/dpo_model"
"""

import argparse
import sys

# ALWAYS import unsloth before transformers/peft/trl
import unsloth
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig


def main():
    parser = argparse.ArgumentParser(description="Unsloth DPO Fine-Tuning Pipeline")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature scaling parameter")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="outputs/dpo_model")
    args = parser.parse_args()

    print("🦥 Enabling Unsloth Fast RL Kernels for DPO...")
    PatchFastRL("DPO", FastLanguageModel)

    print(f"🦥 Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    print("🦥 Adding LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=args.r,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("🦥 Loading sample DPO preference dataset (prompt, chosen, rejected)...")
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:300]")

    print("🦥 Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth automatically handles reference model implicitly
        processing_class=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.1,
            beta=args.beta,
            max_prompt_length=512,
            max_length=args.max_seq_length,
            learning_rate=args.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            output_dir=args.output_dir,
        ),
    )

    print("🦥 Starting DPO Fine-Tuning...")
    trainer.train()

    print(f"🦥 Saving DPO adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ DPO fine-tuning complete!")


if __name__ == "__main__":
    main()
