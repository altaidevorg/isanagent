# CLI, Platform Hardware Matrix, `uv` & Colab Bootstrapping Guide

This reference documents environment management with `uv`, Google Colab automated bootstrapping using `uv`, command-line interface (`unsloth-cli.py`) options, hardware accelerator support (NVIDIA CUDA, AMD ROCm, Intel XPU, Apple Silicon MLX), environment variables, and troubleshooting in **Unsloth**.

---

## 1. Fast Package Management with `uv`

`uv` is an extremely fast, Rust-based Python package and virtual environment manager. It is the recommended environment tool for Unsloth projects across local machines, servers, and cloud notebooks (Google Colab, Kaggle).

### Setting Up a Virtual Environment with `uv`
```bash
# Create a virtual environment with specific Python version
uv venv .venv --python 3.11

# Activate the virtual environment
source .venv/bin/activate  # On Linux / macOS
# .venv\Scripts\activate   # On Windows
```

### PEP 723 Inline Script Metadata
Self-contained scripts can declare their dependencies directly inline using PEP 723 comments. Running `uv run script.py` will resolve and install the exact dependencies automatically into a temporary or managed environment:

```python
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
```

### Installing Unsloth with `uv`

#### CUDA / GPU Setup (Linux / Windows)
```bash
# Install PyTorch with CUDA 12.4 index
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Unsloth and Unsloth Zoo
uv pip install unsloth unsloth_zoo

# Force-reinstall/upgrade to latest un-cached build
uv pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

#### Apple Silicon (MLX Backend) Setup
```bash
# Install Unsloth with MLX support on macOS
uv pip install unsloth unsloth_zoo mlx mlx-lm
```

---

## 2. Google Colab & Cloud Notebook Automated Bootstrapping with `uv`

Standard Google Colab or Kaggle notebook instances do not come with Unsloth pre-installed. Using `uv` inside Colab speeds up dependency resolution by 10x (down to 10-15 seconds instead of 2-3 minutes).

Add this automated `uv` try-except bootstrap block at the very top of Colab scripts to auto-install Unsloth:

```python
# Automated Google Colab Bootstrapping Header with uv
try:
    import unsloth
except ImportError:
    import subprocess
    import sys
    print("🦥 Google Colab environment detected. Bootstrapping Unsloth with uv...")
    # 1. Install uv for 10x faster package installation
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    # 2. Use uv pip install --system for fast Colab system environment setup
    subprocess.run(["uv", "pip", "install", "--system", "--no-deps", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)
    subprocess.run(["uv", "pip", "install", "--system", "--no-deps", "unsloth_zoo"], check=True)
    subprocess.run(["uv", "pip", "install", "--system", "trl", "peft", "accelerate", "transformers", "datasets", "bitsandbytes"], check=True)
    import unsloth
```

---

## 3. Unsloth Command-Line Interface (`unsloth-cli.py`)

Unsloth includes a production-grade CLI runner (`unsloth-cli.py`) for automated training and export.

### CLI Flag Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model_name` | `str` | `"unsloth/llama-3-8b"` | Base model repository or checkpoint folder. |
| `--max_seq_length` | `int` | `2048` | Maximum sequence context length. |
| `--load_in_4bit` | `flag` | `False` | Enables 4-bit QLoRA quantization. |
| `--r` | `int` | `16` | LoRA rank parameter. |
| `--lora_alpha` | `int` | `16` | LoRA alpha scaling parameter. |
| `--lora_dropout` | `float` | `0.0` | LoRA dropout rate. |
| `--use_rslora` | `flag` | `False` | Enables Rank-Stabilized LoRA. |
| `--per_device_train_batch_size` | `int` | `2` | Batch size per GPU device. |
| `--gradient_accumulation_steps` | `int` | `4` | Gradient accumulation count. |
| `--learning_rate` | `float` | `2e-4` | Learning rate for AdamW optimizer. |
| `--max_steps` | `int` | `60` | Total training steps. |
| `--optim` | `str` | `"adamw_8bit"` | Optimizer choice (`"adamw_8bit"`, `"paged_adamw_8bit"`). |
| `--save_model` | `flag` | `False` | Triggers model saving upon completion. |
| `--save_path` | `str` | `"outputs"` | Directory path for saved checkpoint. |
| `--save_method` | `str` | `"merged_16bit"` | Save mode (`"merged_16bit"`, `"merged_4bit"`, `"lora"`). |
| `--save_gguf` | `flag` | `False` | Converts model to GGUF format after training. |
| `--quantization` | `str` | `"q4_k_m"` | GGUF quantization algorithm (`"q4_k_m"`, `"q8_0"`, `"f16"`). |
| `--push_model` | `flag` | `False` | Pushes saved model to Hugging Face Hub. |
| `--hub_path` | `str` | `None` | Target HF Hub repository ID (`username/repo`). |
| `--hub_token` | `str` | `None` | Hugging Face user access token. |

---

## 4. Platform Hardware Matrix

Unsloth automatically detects the underlying hardware accelerator (`DEVICE_TYPE`):

| Backend | Hardware Devices | Python Backend / Acceleration | Notes / Limitations |
| :--- | :--- | :--- | :--- |
| **CUDA** | NVIDIA GPUs (T4, V100, A100, H100, RTX 30/40/50 series) | PyTorch + Triton custom fused kernels | Full feature support (4-bit QLoRA, FP8, GRPO, vLLM, FlashAttention-2). |
| **ROCm** | AMD GPUs (MI200, MI300, RX 7000 series) | PyTorch ROCm + Triton AMD port | Supports 4-bit QLoRA, bitsandbytes ROCm patches, and SFT. |
| **XPU** | Intel GPUs (Data Center GPU Max, Arc) | PyTorch XPU + Intel Triton | Supports 16-bit, 8-bit, and 4-bit SFT. |
| **MLX** | Apple Silicon Macs (M1/M2/M3/M4 Max/Ultra) | `unsloth_zoo.mlx` + Apple MLX framework | Torch-free path. Supports SFT, FastMLXModel, and GGUF export. GRPO/DPO not available on MLX yet. |

---

## 5. Environment Variables Control

| Environment Variable | Allowed Values | Purpose / Effect |
| :--- | :--- | :--- |
| `UNSLOTH_IS_PRESENT` | `"1"` | Automatically set by Unsloth to notify ecosystem packages of optimizations. |
| `UNSLOTH_VLLM_NO_FLASHINFER` | `"1"`, `"0"` | Disables FlashInfer sampling kernels for vLLM on Tesla T4 / Turing GPUs. |
| `UNSLOTH_FORCE_GPU_PATH` | `"1"`, `"0"` | Forces PyTorch/CUDA path on Apple Silicon instead of MLX backend. |
| `UNSLOTH_ENABLE_FULL_FINETUNING` | `"1"`, `"0"` | Enables experimental full-parameter fine-tuning mode. |
| `UNSLOTH_USE_NEW_MODEL` | `"1"`, `"0"` | Toggles updated `FastBaseModel` architecture pipeline. |
| `UNSLOTH_ALLOW_CPU` | `"1"`, `"0"` | CPU-only CI mode. Disables CUDA kernel initialization checks. |
| `UNSLOTH_DISABLE_AUTO_UPDATES` | `"1"`, `"0"` | Suppresses automated `unsloth_zoo` upgrade prompts. |
| `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` | `"1"` | Relaxes Metal timeout on macOS to prevent MLX context timeouts. |
