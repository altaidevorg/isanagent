#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth>=2025.2.1",
#     "unsloth_zoo>=2025.2.1",
#     "torch>=2.4.0",
#     "transformers>=4.48.0",
#     "peft>=0.14.0",
#     "trl>=0.14.0",
#     "datasets>=3.2.0",
#     "accelerate>=1.3.0",
# ]
# ///
"""
🦥 Unsloth Supervised Fine-Tuning (SFT) Template Script

Usage:
    python finetune_sft.py \
        --model_name "unsloth/Qwen3.5-9B-Instruct" \
        --max_seq_length 4096 \
        --output_dir "outputs/sft_model"
"""

import argparse
import os
import sys

# ALWAYS import unsloth before transformers/peft/trl
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="Unsloth SFT Fine-Tuning Pipeline")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3.5-9B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--override_chat_template", type=str, default=None, help="Optional fallback template name if tokenizer.chat_template is missing")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True, help="Enable 4-bit NF4 QLoRA quantization")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="outputs/sft_model")
    args = parser.parse_args()

    print(f"🦥 Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
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
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # 0.0 enables Triton fused kernels; use >0.0 (e.g. 0.05) if overfitting
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Use native chat_template if present; fallback to get_chat_template only if requested/missing
    if args.override_chat_template:
        tokenizer = get_chat_template(tokenizer, chat_template=args.override_chat_template)
    elif not getattr(tokenizer, "chat_template", None):
        print("⚠️ Model tokenizer lacks native chat_template, falling back to chatml template...")
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    print("🦥 Preparing sample dataset...")
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

    dataset = dataset.map(format_prompts, batched=True)

    print("🦥 Initializing SFTTrainer with current TRL SFTConfig signature...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.max_seq_length,
            packing=False,  # Set packing=True only with Flash Attention varlen to prevent cross-sample leakage
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
        ),
    )

    # Response-only loss masking for instruction tuning
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("🦥 Starting Training...")
    trainer.train()

    print(f"🦥 Saving fine-tuned LoRA adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Fine-tuning complete!")


if __name__ == "__main__":
    main()
