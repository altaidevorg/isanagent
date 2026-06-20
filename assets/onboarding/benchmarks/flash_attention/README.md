# Simplified FlashAttention forward (Phase 4)

Port a minimal forward-only attention block from PyTorch/Triton reference.

Evolution targets: tile sizes M/N, pipeline depth.

See `kernels/reference/Triton To Pallas Conversion.md` for BlockSpec attention tiling patterns.
