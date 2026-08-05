# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "unsloth_zoo",
#     "trl",
#     "transformers",
#     "peft",
#     "datasets",
#     "accelerate",
#     "torch",
#     "vllm",
# ]
# ///

import os
import re
import sys
import torch

# Dynamic FlashInfer Check: Disable FlashInfer ONLY on older GPUs (Turing / Tesla T4 compute capability <= 7.5)
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    if major < 8:  # Tesla T4, GTX 1660, RTX 2080
        os.environ.setdefault("UNSLOTH_VLLM_NO_FLASHINFER", "1")

import argparse
import unsloth
from unsloth import FastLanguageModel, PatchFastRL
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig


# 1. Reward Functions
def xml_layout_reward_func(completions, **kwargs):
    """Checks if output matches <think>...</think><answer>...</answer> XML tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        rewards.append(1.0 if re.match(pattern, text, re.DOTALL) else 0.0)
    return rewards


def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Extracts answer inside <answer>...</answer> and compares to ground truth."""
    rewards = []
    for completion, target in zip(completions, answer):
        text = completion[0]["content"]
        extracted = text.split("<answer>")[-1].split("</answer>")[0].strip() if "<answer>" in text else ""
        rewards.append(2.0 if extracted == str(target).strip() else 0.0)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Unsloth GRPO Reasoning RL Pipeline")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="outputs/grpo_reasoning")
    args = parser.parse_args()

    print("🦥 Enabling Unsloth Fast RL Kernels for GRPO...")
    PatchFastRL("GRPO", FastLanguageModel)

    # Dynamic VRAM Allocation: Use 0.4 for 16GB VRAM GPUs (e.g. T4), scale up for larger GPUs (A100/H100)
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 16
    vllm_mem_util = 0.4 if gpu_vram_gb <= 16 else 0.7

    print(f"🦥 Loading base reasoning model: {args.model_name} (vLLM memory utilization: {vllm_mem_util})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,              # Boot vLLM sampling engine
        gpu_memory_utilization=vllm_mem_util, # Dynamic VRAM allocation
    )

    print("🦥 Applying LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
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

    print("🦥 Preparing GSM8K Math Reasoning Dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train[:300]")

    def format_dataset(example):
        question = example["question"]
        # Extract ground truth answer from GSM8K format '#### 42'
        answer = example["answer"].split("####")[-1].strip()
        prompt = [
            {
                "role": "system",
                "content": "Respond in the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>",
            },
            {"role": "user", "content": question},
        ]
        return {
            "prompt": prompt,
            "answer": answer,
        }

    dataset = dataset.map(format_dataset)

    print("🦥 Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[xml_layout_reward_func, correctness_reward_func],
        args=GRPOConfig(
            use_vllm=True,
            vllm_gpu_memory_utilization=vllm_mem_util,  # Dynamic VRAM allocation
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=8,
            max_prompt_length=512,
            max_completion_length=1024,
            output_dir=args.output_dir,
        ),
        train_dataset=dataset,
    )

    print("🦥 Starting GRPO Reasoning Training...")
    trainer.train()

    print(f"🦥 Saving GRPO reasoning model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ GRPO reasoning training complete!")


if __name__ == "__main__":
    main()
