# Bumblebee Quantized Inference Plan

**Target:** `elixir-nx/bumblebee`
**Goal:** Enable off-the-shelf 4-bit quantized LLM inference with LoRA adapters

---

## The Vision

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

Nx.Serving.batched_run(serving, "Write a post in your style")
```

---

## Required Changes

### 1. Quantized Model Loading

**File:** `lib/bumblebee/loader.ex` (or new `lib/bumblebee/quantized_loader.ex`)

Add to `Bumblebee.load_model/2`:
- `quantization: :int4` option
- Load quantized weights as triplets (weight, scales, biases)
- Store quantization metadata in model struct

```elixir
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

**Key insight:** Quantized models store weights as triplets:
- `layer.weight` - uint32 packed int4 values
- `layer.scales` - bfloat16 scale factors (1 per 64 weights)
- `layer.biases` - bfloat16 zero points (1 per 64 weights)

### 2. Quantized Axon Layers

**File:** `lib/bumblebee/layers.ex` (or integrate with existing layers)

New layer type for quantized dense:

```elixir
defn quantized_dense(input, weight, scales, biases, opts \\ []) do
  group_size = opts[:group_size] || 64
  bits = opts[:bits] || 4
  EMLX.quantized_matmul(input, weight, scales, biases, true, group_size, bits)
end
```

**Challenge:** This is EMLX-specific. Options:
1. Add to EMLX as backend-specific layer
2. Add to Nx as `Nx.LinAlg.quantized_matmul` with backend dispatch
3. Keep in Bumblebee but require EMLX backend for quantized models

### 3. Quantized Model Definitions

**Files:** `lib/bumblebee/text/qwen3.ex` (and other model defs)

Modify model definitions to use quantized ops conditionally:

```elixir
defp dense_layer(input, params, opts) do
  if opts[:quantized] do
    quantized_dense(input, params.weight, params.scales, params.biases, opts)
  else
    Axon.Layers.dense(input, params.kernel, params.bias)
  end
end
```

### 4. Runtime LoRA Support

**File:** New `lib/bumblebee/adapters.ex`

```elixir
defmodule Bumblebee.Adapters do
  @moduledoc """
  Runtime adapter (LoRA) loading and application.
  """

  def load_adapter(source, opts \\ []) do
    # Load LoRA A/B matrices from safetensors
    # Return adapter struct with scale factor
  end

  defn apply_lora(output, input, adapter, scaling) do
    lora_out = input
      |> Nx.dot(adapter.lora_a)
      |> Nx.dot(adapter.lora_b)
      |> Nx.multiply(scaling)
    Nx.add(output, lora_out)
  end
end
```

**LoRA math:** `y = Wx + scale * (x @ A @ B)`
- A: [in_features, rank] - typically rank=8
- B: [rank, out_features]
- scale: typically 16.0-32.0 for strong adapter influence

---

## Implementation Order

### Phase 1: Foundation (separate from Bumblebee)
- [x] EMLX quantization NIFs (done, PR #95 pending)
- [ ] Safetensors Hex package

### Phase 2: Loading
- [ ] Add safetensors support to Bumblebee
- [ ] Add quantized model loading option
- [ ] Handle weight triplets (weight/scales/biases)

### Phase 3: Inference
- [ ] Add quantized dense layer
- [ ] Modify Qwen3 model def for quantized path
- [ ] Test with Qwen3-8B-4bit

### Phase 4: LoRA
- [ ] Add Bumblebee.load_adapter/2
- [ ] Add runtime LoRA application
- [ ] Document training workflow

---

## Design Decisions

1. **Backend specificity:** EMLX-only (Apple Silicon)
   - quantized_matmul is an EMLX-specific operation
   - Requires Apple Silicon unified memory architecture
   - Clear error message if user tries with different backend

2. **Model definition changes:** Separate quantized model definitions
   - New `Bumblebee.Text.Qwen3Quantized` module
   - Configurable via options
   - Doesn't pollute the standard model definitions

3. **LoRA scope:** Both inference AND training
   - `Bumblebee.load_adapter/2` - load pre-trained adapters
   - `Bumblebee.train_adapter/3` - train new adapters
   - Full workflow: prepare data → train → save → load → inference

---

## Reference Implementation

The working code is in bobby_posts:
- `lib/bobby_posts/quantized_loader.ex` - loads quantized models
- `lib/bobby_posts/adapter_loader.ex` - loads LoRA adapters
- `lib/bobby_posts/qwen3/` - custom Qwen3 inference (attention, model, layers)
- `lib/bobby_posts/safetensors.ex` - safetensors parser

Total: ~2000 lines of Elixir that could be extracted/adapted for Bumblebee.
