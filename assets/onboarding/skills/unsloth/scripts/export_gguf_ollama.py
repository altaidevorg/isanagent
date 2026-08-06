#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth>=2025.2.1",
#     "unsloth_zoo>=2025.2.1",
#     "torch>=2.4.0",
#     "transformers>=4.48.0",
#     "peft>=0.14.0",
#     "huggingface-hub>=0.20.0",
# ]
# ///
"""
🦥 Unsloth Model Export Utility (GGUF, Merged Adapters, Ollama, HF Hub)

Usage:
    # Convert and save GGUF locally
    python export_gguf_ollama.py --model_path "outputs/sft_model" --save_gguf --quant_method "q4_k_m"

    # Merge adapters into 16-bit and push to Hugging Face Hub
    python export_gguf_ollama.py --model_path "outputs/sft_model" --push_merged --hub_path "username/my-model" --hub_token "hf_xxx"
"""

import argparse
import sys
import unsloth
from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Unsloth Model Export & Quantization Utility")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LoRA checkpoint or base model")
    parser.add_argument("--save_gguf", action="store_true", help="Convert model to GGUF format locally")
    parser.add_argument("--quant_method", type=str, default="q4_k_m", help="Quantization algorithm (q4_k_m, q8_0, f16, q5_k_m)")
    parser.add_argument("--output_path", type=str, default="outputs/exported_model")
    parser.add_argument("--save_merged", action="store_true", help="Save merged 16bit model locally")
    parser.add_argument("--push_gguf", action="store_true", help="Push GGUF + Ollama Modelfile to HF Hub")
    parser.add_argument("--push_merged", action="store_true", help="Push merged 16bit model to HF Hub")
    parser.add_argument("--hub_path", type=str, default=None, help="HF Hub repository ID (e.g. username/repo)")
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face access token")
    args = parser.parse_args()

    print(f"🦥 Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    if args.save_merged:
        print(f"🦥 Merging adapters and saving 16-bit model to {args.output_path}_16bit...")
        model.save_pretrained_merged(f"{args.output_path}_16bit", tokenizer, save_method="merged_16bit")
        print("✅ Merged 16-bit model saved.")

    if args.save_gguf:
        print(f"🦥 Quantizing and exporting GGUF ({args.quant_method}) to {args.output_path}_gguf...")
        model.save_pretrained_gguf(
            f"{args.output_path}_gguf",
            tokenizer,
            quantization_method=args.quant_method,
        )
        print("✅ GGUF export complete.")

    if args.push_gguf:
        if not args.hub_path:
            raise ValueError("--hub_path is required when using --push_gguf")
        print(f"🦥 Pushing GGUF ({args.quant_method}) & Ollama Modelfile to HF Hub repository: {args.hub_path}...")
        model.push_to_hub_gguf(
            args.hub_path,
            tokenizer,
            quantization_method=args.quant_method,
            token=args.hub_token,
        )
        print(f"✅ Successfully pushed to HF Hub! Run with Ollama: ollama run hf.co/{args.hub_path}:{args.quant_method.upper()}")

    if args.push_merged:
        if not args.hub_path:
            raise ValueError("--hub_path is required when using --push_merged")
        print(f"🦥 Pushing merged 16-bit model to HF Hub repository: {args.hub_path}...")
        model.push_to_hub_merged(
            args.hub_path,
            tokenizer,
            save_method="merged_16bit",
            token=args.hub_token,
        )
        print("✅ Successfully pushed merged model to HF Hub!")


if __name__ == "__main__":
    main()
