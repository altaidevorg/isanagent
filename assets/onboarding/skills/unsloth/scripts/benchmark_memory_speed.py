#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth>=2026.8.0",
#     "unsloth_zoo>=2026.8.0",
#     "torch>=2.13.0",
#     "transformers>=5.14.1",
#     "peft>=0.20.0",
#     "accelerate>=1.14.0",
# ]
# ///
"""
🦥 Unsloth VRAM Memory & Speed Profiling Benchmark Tool

Usage:
    python benchmark_memory_speed.py --model_name "unsloth/Qwen3-8B"
"""

import argparse
import sys
import time

import torch
import unsloth
from unsloth import FastLanguageModel


def get_memory_stats():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        return gpu_name, total_vram_gb, allocated_gb, reserved_gb
    return "CPU", 0.0, 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description="Unsloth VRAM & Inference Speed Benchmark")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-8B")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms:")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print("📊 Measuring Initial GPU Memory...")
    gpu_name, total_vram, alloc_mem, res_mem = get_memory_stats()
    print(f"  GPU Name: {gpu_name}")
    print(f"  Total VRAM: {total_vram:.2f} GB")
    print(f"  Initial Allocated VRAM: {alloc_mem:.2f} GB")

    print(f"\n🦥 Loading Model in 4-bit: {args.model_name}...")
    start_load = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    load_time = time.time() - start_load
    print(f"  Model Loaded in {load_time:.2f} seconds.")

    _, _, post_load_alloc, _ = get_memory_stats()
    print(f"  VRAM allocated for model: {post_load_alloc:.2f} GB")

    print("\n🦥 Enabling Fast Inference Mode...")
    FastLanguageModel.for_inference(model)

    inputs = tokenizer([args.prompt], return_tensors="pt")
    if hasattr(inputs, "to"):
        try:
            inputs = inputs.to("cuda")
        except Exception:
            pass

    print(f"\n⚡ Running Generation ({args.max_new_tokens} tokens)...")
    start_gen = time.time()
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, use_cache=True)
    gen_time = time.time() - start_gen

    num_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
    tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0

    print(f"  Generated {num_tokens} tokens in {gen_time:.2f} seconds.")
    print(f"  Speed: {tokens_per_sec:.2f} tokens/second")

    _, _, final_alloc, _ = get_memory_stats()
    print(f"  VRAM allocated after generation: {final_alloc:.2f} GB")


if __name__ == "__main__":
    main()
