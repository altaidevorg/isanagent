#!/usr/bin/env python3
"""
🦥 Unsloth Supervised Fine-Tuning (SFT) Template Script

Usage:
    python finetune_sft.py \
        --model_name "unsloth/Qwen2.5-7B-Instruct" \
        --max_seq_length 4096 \
        --chat_template "qwen-2.5" \
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
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--chat_template", type=str, default="qwen-2.5")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
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
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    print("🦥 Preparing sample dataset...")
    # Load sample dataset (or replace with your custom dataset)
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

    print("🦥 Initializing SFTTrainer with response-only loss masking...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=SFTConfig(
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

    # Response-only loss masking
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
