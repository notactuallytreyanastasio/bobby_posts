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

| File | Lines | Description |
|------|-------|-------------|
| `quantized_loader.ex` | 200 | Load quantized models from safetensors |
| `quantized_serving.ex` | 250 | Nx.Serving for quantized text generation |
| `adapters.ex` | 350 | LoRA adapter load/apply/train |
| `models/qwen3_quantized.ex` | 500 | Full Qwen3 quantized model definition |
| `training.ex` | 300 | LoRA training workflow integration |

**Total: ~1,600 lines of contribution-ready Elixir code**

## Dependencies

- `emlx` with quantization ops (PR #95: https://github.com/elixir-nx/emlx/pull/95)
- `safetensors` Hex package (https://github.com/notactuallytreyanastasio/safetensors_ex)

## Full Workflow

```elixir
# 1. Prepare training data
Bumblebee.Training.LoRA.prepare_data(my_posts, "/path/to/data",
  prompt: "Write a post in my style",
  min_length: 160
)

# 2. Train adapter (calls Python mlx_lm)
{:ok, adapter_path} = Bumblebee.Training.LoRA.train(
  base_model: "lmstudio-community/Qwen3-8B-MLX-4bit",
  training_data: "/path/to/data",
  output_path: "/path/to/adapter",
  iterations: 25_000,
  rank: 8,
  scale: 20.0
)

# 3. Load model, adapter, tokenizer
{:ok, model} = Bumblebee.QuantizedLoader.load_model("/path/to/Qwen3-8B-4bit")
{:ok, adapter} = Bumblebee.Adapters.load(adapter_path)
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-8B"})

# 4. Create serving and generate
serving = Bumblebee.Text.QuantizedGeneration.new(model, tokenizer,
  adapter: adapter,
  max_new_tokens: 100,
  temperature: 0.8
)

Nx.Serving.run(serving, "Write a post")
```

## Performance

Tested on Apple Silicon:

| Metric | Value |
|--------|-------|
| Model Load Time | 4-6 seconds |
| Single Token Latency | ~7ms (135 tok/s) |
| Generation Throughput | ~21 tok/s |
| Memory Usage | ~5GB for Qwen3-8B-4bit |
