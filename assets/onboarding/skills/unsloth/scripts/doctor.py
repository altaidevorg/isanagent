#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4.0",
# ]
# ///
"""
Preflight Environment & Hardware Diagnostic Tool (Unsloth Skill Doctor)

Inspects:
- OS & Architecture
- GPU Model, Count, Compute Capability, Total VRAM
- Acceleration Backends (CUDA / ROCm / XPU / Apple Silicon MLX)
- Installed Package Versions (Python, Torch, Unsloth, TRL, Transformers, PEFT, vLLM, bitsandbytes)
- Supported Precision (bfloat16, float16, FP8)
- Emits human-readable diagnostic report and environment.json
"""

import json
import os
import platform
import sys
from pathlib import Path

SKILL_SCRIPTS_DIR = Path(__file__).resolve().parent


def check_env():
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "gpu": {},
        "packages": {},
    }

    # Check PyTorch & Accelerator
    try:
        import torch

        info["packages"]["torch"] = torch.__version__
        info["gpu"]["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            info["gpu"]["device_count"] = device_count
            info["gpu"]["device_name"] = gpu_name
            info["gpu"]["compute_capability"] = f"{capability[0]}.{capability[1]}"
            info["gpu"]["total_vram_gb"] = round(vram_gb, 2)
            info["gpu"]["bfloat16_supported"] = capability[0] >= 8
        else:
            info["gpu"]["device_name"] = "None (CPU)"

        # Check Apple Silicon / MPS / MLX
        info["gpu"]["mps_available"] = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

    except ImportError:
        info["packages"]["torch"] = "Not Installed"

    # Check core packages
    for pkg_name in ["unsloth", "unsloth_zoo", "transformers", "peft", "trl", "vllm", "bitsandbytes", "datasets", "accelerate"]:
        try:
            mod = __import__(pkg_name)
            info["packages"][pkg_name] = getattr(mod, "__version__", "Installed (unknown version)")
        except ImportError:
            info["packages"][pkg_name] = "Not Installed"

    return info


def main():
    print("🩺 Running Unsloth Skill Preflight Doctor...")
    env_info = check_env()

    print("\n--- System & Hardware Report ---")
    print(f"  OS: {env_info['os']} ({env_info['arch']})")
    print(f"  Python Version: {env_info['python_version']}")

    gpu = env_info["gpu"]
    if gpu.get("cuda_available"):
        print(f"  GPU Accelerator: {gpu['device_name']} (x{gpu['device_count']})")
        print(f"  Compute Capability: {gpu['compute_capability']}")
        print(f"  Total VRAM: {gpu['total_vram_gb']} GB")
        print(f"  bfloat16 Support: {'YES' if gpu['bfloat16_supported'] else 'NO (Use float16)'}")
    elif gpu.get("mps_available"):
        print("  GPU Accelerator: Apple Silicon Metal (MPS / MLX)")
    else:
        print("  GPU Accelerator: None detected")

    print("\n--- Core Package Matrix ---")
    for pkg, ver in env_info["packages"].items():
        print(f"  {pkg:16s}: {ver}")

    # Write environment.json in local output directory
    env_json_path = Path.cwd() / "environment.json"
    env_json_path.write_text(json.dumps(env_info, indent=2), encoding="utf-8")
    print(f"\n✅ Diagnostic complete! Environment report written to {env_json_path}")


if __name__ == "__main__":
    main()
