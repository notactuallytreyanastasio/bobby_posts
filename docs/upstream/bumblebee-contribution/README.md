# Bumblebee Quantized LLM Contribution

This directory contains code and documentation for contributing quantized LLM inference to Bumblebee.

## Scope

- **4-bit quantized model loading** (MLX format)
- **Quantized inference** using EMLX backend
- **Runtime LoRA adapters** (load and apply)
- **LoRA training** via mlx_lm integration

## Target: Apple Silicon Only

This uses EMLX (MLX backend for Nx) which only works on Apple Silicon Macs. The quantized_matmul operation leverages MLX's fused int4 kernels optimized for unified memory.

## Files

- `quantized_loader.ex` - Load quantized models from safetensors
- `quantized_serving.ex` - Nx.Serving for quantized text generation
- `adapters.ex` - LoRA adapter loading and application
- `models/qwen3_quantized.ex` - Qwen3 quantized model definition
- `training.ex` - LoRA training integration

## Dependencies

- `emlx` with quantization ops (PR #95)
- `safetensors` Hex package
