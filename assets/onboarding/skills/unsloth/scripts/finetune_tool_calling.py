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
🦥 Unsloth Tool-Calling / Agent SFT Fine-Tuning Script

Usage:
    python finetune_tool_calling.py \
        --model_name "unsloth/Qwen3.5-9B-Instruct" \
        --output_dir "outputs/tool_calling_model"
"""

import argparse
import sys

# ALWAYS import unsloth before transformers/peft/trl
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="Unsloth Tool-Calling Fine-Tuning Pipeline")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3.5-9B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True, help="Enable 4-bit NF4 QLoRA quantization")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="outputs/tool_calling_model")
    args = parser.parse_args()

    print(f"🦥 Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    print("🦥 Adding LoRA Adapters for Tool Calling...")
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

    print("🦥 Preparing Tool-Calling Sample Dataset...")
    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name, e.g. London"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    sample_conversations = [
        [
            {"role": "user", "content": "What is the weather like in Tokyo right now?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_current_weather", "arguments": '{"location": "Tokyo", "unit": "celsius"}'},
                    }
                ],
            },
            {"role": "tool", "name": "get_current_weather", "content": '{"temperature": 22, "condition": "Sunny"}'},
            {"role": "assistant", "content": "The weather in Tokyo is currently sunny with a temperature of 22°C."},
        ]
    ] * 50

    dataset = Dataset.from_dict({"messages": sample_conversations})

    def format_tool_prompts(examples):
        texts = []
        for convo in examples["messages"]:
            formatted = tokenizer.apply_chat_template(
                convo,
                tools=tools_definition,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(formatted)
        return {"text": texts}

    formatted_dataset = dataset.map(format_tool_prompts, batched=True)

    print("🦥 Initializing SFTTrainer for Tool-Calling Fine-Tuning...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.max_seq_length,
            packing=False,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir=args.output_dir,
        ),
    )

    print("🦥 Starting Tool-Calling Fine-Tuning...")
    trainer.train()

    print(f"🦥 Saving fine-tuned tool-calling adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Tool-calling fine-tuning complete!")


if __name__ == "__main__":
    main()
