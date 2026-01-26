# Bumblebee Integration Plan

## Goal: Off-the-Shelf Quantized LoRA Inference

Enable other developers to do this:

```elixir
# Future ideal API
{:ok, model} = Bumblebee.load_model(
  {:hf, "lmstudio-community/Qwen3-8B-MLX-4bit"},
  quantization: :int4,
  backend: {EMLX.Backend, device: :gpu}
)

{:ok, adapter} = Bumblebee.load_adapter(
  {:hf, "username/my-fine-tuned-adapter"},
  scale: 20.0
)

{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-8B"})

serving = Bumblebee.Text.generation(model, tokenizer,
  adapter: adapter,
  compile: [batch_size: 1, sequence_length: 256]
)

# Generate text
Nx.Serving.batched_run(serving, "Write a post in your style")
```

## Required Upstreams

### 1. EMLX: Quantization NIFs
**PR to:** `elixir-nx/emlx`
**Status:** Fork ready at `notactuallytreyanastasio/emlx` branch `feat/quantization-ops`

Add to EMLX:
- `EMLX.quantized_matmul/7` - fused 4-bit matmul
- `EMLX.dequantize/5` - expand int4 to float
- `EMLX.quantize/3` - compress float to int4

```cpp
// The ~60 lines of C++ we wrote
NIF(quantized_matmul) {
    TENSOR_PARAM(0, x);
    TENSOR_PARAM(1, w);
    TENSOR_PARAM(2, scales);
    TENSOR_PARAM(3, biases);
    PARAM(4, bool, transpose);
    PARAM(5, int, group_size);
    PARAM(6, int, bits);
    TENSOR(mlx::core::quantized_matmul(*x, *w, *scales, *biases, transpose, group_size, bits, device));
}
```

**Effort:** Small - code is written and tested

### 2. Safetensors: Hex Package
**PR to:** New package or integrate into Nx
**Status:** Package ready at `notactuallytreyanastasio/safetensors_ex`

Pure Elixir safetensors parser:
- Parse HuggingFace `.safetensors` format
- Load tensors directly into Nx
- Handle quantized dtypes (uint32 packed int4)

**Effort:** Small - code is written, needs polish for Hex

### 3. Bumblebee: Quantized Model Loading
**PR to:** `elixir-nx/bumblebee`

Add to `Bumblebee.load_model/2`:
- `quantization: :int4` option
- Load quantized weights as triplets (weight, scales, biases)
- Store quantization metadata in model struct

```elixir
# New loader code needed
defp load_quantized_params(path, opts) do
  tensors = Safetensors.load(path, backend: {EMLX.Backend, device: :gpu})

  # Group into quantized layers
  for {name, weight} <- tensors, String.ends_with?(name, ".weight") do
    scales = tensors[String.replace(name, ".weight", ".scales")]
    biases = tensors[String.replace(name, ".weight", ".biases")]
    {name, %QuantizedLinear{weight: weight, scales: scales, biases: biases}}
  end
end
```

**Effort:** Medium - needs careful integration with existing loaders

### 4. Bumblebee: Quantized Inference
**PR to:** `elixir-nx/bumblebee`

Modify `Axon` model definitions to use quantized ops:
- New `Axon.quantized_dense/3` layer type
- Uses `EMLX.quantized_matmul` instead of `Nx.dot`
- Conditional based on model quantization metadata

```elixir
# New Axon layer
defn quantized_dense(input, weight, scales, biases, opts \\ []) do
  group_size = opts[:group_size] || 64
  bits = opts[:bits] || 4
  EMLX.quantized_matmul(input, weight, scales, biases, true, group_size, bits)
end
```

**Effort:** Medium-High - touches core inference path

### 5. Bumblebee: Runtime LoRA
**PR to:** `elixir-nx/bumblebee`

Add adapter support:
- `Bumblebee.load_adapter/2` function
- Runtime LoRA application in forward pass
- Support for multiple adapter formats

```elixir
# New LoRA application
defn apply_lora(output, input, adapter, scaling) do
  lora_out = input
    |> Nx.dot(adapter.lora_a)
    |> Nx.dot(adapter.lora_b)
    |> Nx.multiply(scaling)
  Nx.add(output, lora_out)
end
```

**Effort:** Medium - new feature, well-defined scope

### 6. Documentation: Training Guide

Create documentation for:
- Training data format (ChatML JSONL)
- LoRA training with `python -m mlx_lm.lora`
- Recommended hyperparameters
- Adapter format specification

## Phased Approach

### Phase 1: Foundation (EMLX + Safetensors)
1. PR EMLX quantization ops
2. Publish safetensors_ex to Hex
3. Write docs for both

**Outcome:** Developers can manually load quantized models

### Phase 2: Loading (Bumblebee Loader)
1. Add quantized model loading to Bumblebee
2. Integrate safetensors parser
3. Add tests for various quantized models

**Outcome:** `Bumblebee.load_model` works with 4-bit models

### Phase 3: Inference (Bumblebee Serving)
1. Add quantized Axon layers
2. Modify Qwen3 model definition for quantized path
3. Benchmark and optimize

**Outcome:** `Bumblebee.Text.generation` works with quantized models

### Phase 4: Fine-tuning (LoRA Support)
1. Add adapter loading
2. Add runtime LoRA application
3. Document training workflow

**Outcome:** Full fine-tuning workflow in Elixir ecosystem

## Training Data Format

For other developers to train their own adapters:

```jsonl
{"messages": [{"role": "user", "content": "Write a post"}, {"role": "assistant", "content": "Your post here"}]}
{"messages": [{"role": "user", "content": "Share a thought"}, {"role": "assistant", "content": "Another post"}]}
```

Training command:
```bash
# Create a config file (e.g., training_config.yaml):
# model: lmstudio-community/Qwen3-8B-MLX-4bit
# data: ./your_posts
# train: true
# iters: 25000
# learning_rate: 1e-5
# mask_prompt: true
# lora_parameters:
#   rank: 8
#   scale: 20.0

python -m mlx_lm.lora --config training_config.yaml
```

## Who Benefits

1. **Elixir developers** wanting local LLM inference on Mac
2. **Fine-tuning enthusiasts** who prefer Elixir over Python
3. **Privacy-conscious users** who want on-device inference
4. **Phoenix/LiveView apps** needing AI features without external APIs

## Resources

- EMLX fork: https://github.com/notactuallytreyanastasio/emlx/tree/feat/quantization-ops
- Safetensors: https://github.com/notactuallytreyanastasio/safetensors_ex
- This project: https://github.com/notactuallytreyanastasio/bobby_posts
- Adapters: https://github.com/notactuallytreyanastasio/bobby_posts_adapters
