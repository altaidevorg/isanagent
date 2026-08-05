#!/usr/bin/env python3
"""
🦥 Unsloth VRAM Memory & Speed Profiling Benchmark Tool

Usage:
    python benchmark_memory_speed.py --model_name "unsloth/Qwen2.5-7B-Instruct"
"""

import argparse
import sys
import time

import unsloth
from unsloth import FastLanguageModel, get_gpu_memory_stats


def main():
    parser = argparse.ArgumentParser(description="Unsloth VRAM & Inference Speed Benchmark")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms:")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print("📊 Measuring Initial GPU Memory...")
    gpu_stats, initial_peak_gb, max_memory_gb = get_gpu_memory_stats()
    print(f"  GPU Name: {gpu_stats.name}")
    print(f"  Total VRAM: {max_memory_gb:.2f} GB")

    print(f"\n🦥 Loading Model in 4-bit: {args.model_name}...")
    start_load = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    load_time = time.time() - start_load
    print(f"  Model Loaded in {load_time:.2f} seconds.")

    _, post_load_peak_gb, _ = get_gpu_memory_stats()
    print(f"  VRAM allocated for model: {post_load_peak_gb:.2f} GB")

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

    _, final_peak_gb, _ = get_gpu_memory_stats()
    print(f"  Peak VRAM used during generation: {final_peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
