#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Training Run Planner Tool (plan_run.py)

Given model size, target sequence length, GPU VRAM, and training type (SFT, DPO, GRPO),
this tool calculates:
- Batch size and gradient accumulation steps
- Enforces GRPO batch divisibility: (batch_size * grad_accum) % num_generations == 0
- Recommends precision, checkpointing, and optimizer settings
- Generates plan_run.json config
"""

import argparse
import json
import sys
from pathlib import Path


def plan_training_run(model_name, workflow, vram_gb, max_seq_length, num_generations=8):
    plan = {
        "model_name": model_name,
        "workflow": workflow,
        "max_seq_length": max_seq_length,
        "vram_gb": vram_gb,
    }

    if workflow == "grpo":
        if vram_gb <= 16:
            plan["batch_size"] = 1
            plan["grad_accum"] = 8
            plan["vllm_mem_util"] = 0.4
        else:
            plan["batch_size"] = 2
            plan["grad_accum"] = 4
            plan["vllm_mem_util"] = 0.7

        plan["num_generations"] = num_generations
        effective_batch = plan["batch_size"] * plan["grad_accum"]
        assert effective_batch % num_generations == 0, f"GRPO batch size error: {effective_batch} not divisible by {num_generations}"
        plan["divisibility_check"] = f"PASSED ({effective_batch} % {num_generations} == 0)"
    else:
        if vram_gb <= 16:
            plan["batch_size"] = 1
            plan["grad_accum"] = 8
        else:
            plan["batch_size"] = 2
            plan["grad_accum"] = 4

    plan["load_in_4bit"] = True
    plan["lora_r"] = 16
    plan["lora_dropout"] = 0.0
    plan["gradient_checkpointing"] = "unsloth"
    plan["optim"] = "adamw_8bit"

    return plan


def main():
    parser = argparse.ArgumentParser(description="Unsloth Training Run Planner")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3.5-9B-Instruct")
    parser.add_argument("--workflow", choices=["sft", "dpo", "grpo"], default="sft")
    parser.add_argument("--vram_gb", type=float, default=16.0)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--output", type=str, default="plan_run.json")
    args = parser.parse_args()

    print(f"📋 Planning training run for {args.model_name} ({args.workflow.upper()})...")
    plan = plan_training_run(
        model_name=args.model_name,
        workflow=args.workflow,
        vram_gb=args.vram_gb,
        max_seq_length=args.max_seq_length,
        num_generations=args.num_generations,
    )

    print("\n--- Recommended Execution Plan ---")
    for k, v in plan.items():
        print(f"  {k:24s}: {v}")

    output_path = Path.cwd() / args.output
    output_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"\n✅ Execution plan written to {output_path}")


if __name__ == "__main__":
    main()
