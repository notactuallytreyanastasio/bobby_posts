# From Python to Pure Elixir: Building a LoRA-Enabled LLM Stack for Apple Silicon

*How we built a complete LLM inference pipeline in Elixir, eliminating Python dependencies while maintaining full fine-tuning fidelity.*

## The Problem

The Python ML ecosystem is undeniably powerful. PyTorch, HuggingFace Transformers, and MLX make it trivial to run large language models. But for those of us building production systems in Elixir, shelling out to Python feels wrong. It's a deployment headache, a performance bottleneck, and an architectural smell.

Elixir has Nx (numerical computing) and Axon (neural networks), but the ecosystem lacks:
- **Quantized model support** - 4-bit inference is essential for running 8B+ models on consumer hardware
- **Safetensors parsing** - HuggingFace's standard model format has no Elixir parser
- **LoRA adapter loading** - Fine-tuned adapters need to be applied at runtime

Our goal: Run a fine-tuned Qwen3-8B model on Apple Silicon, generating tweets in a specific voice, with **zero Python at runtime**.

## The Journey

This project evolved through several iterations, tracked in our decision graph:

```
Node 1: Implement Bluesky Bot (Python - it worked)
  ↓
Node 19: Build in Elixir/Phoenix (the dream)
  ↓
Node 32: Created bobby_posts app
  ↓
Node 35: Upstream Pure Elixir LoRA Stack
  ↓
Node 36: v1.0.0 tagged - pure Elixir achieved
```

### Phase 1: Python Worked, But...

The Python bot was simple:

```python
from mlx_lm import load, generate

model, tokenizer = load("Qwen3-8B-MLX-4bit", adapter_path="adapters")
response = generate(model, tokenizer, prompt="Write a tweet...")
```

It ran at ~21 tok/s on an M-series Mac. But every generation spawned a Python process, loaded 5GB of weights, and required maintaining a separate Python environment. Not ideal.

### Phase 2: Discovering EMLX

EMLX is the MLX backend for Nx, providing GPU acceleration on Apple Silicon. But it lacked quantization support - the critical feature for running large models in reasonable memory.

We forked EMLX and added three NIFs:

```cpp
// c_src/emlx_nif.cpp
{"quantized_matmul", 8, quantized_matmul},  // Fused 4-bit matmul
{"dequantize", 6, dequantize},              // Unpack weights
{"quantize", 4, quantize}                    // Pack weights
```

These wrap MLX's C++ quantization kernels, which compile to Metal shaders at runtime.

### Phase 3: Building the Inference Stack

With quantized operations available, we built the full stack:

```
┌─────────────────────────────────────────────────────────┐
│  BobbyPosts.Generator (GenServer)                       │
│  - Holds loaded model in memory                         │
│  - Manages tokenizer                                    │
│  - Orchestrates generation                              │
├─────────────────────────────────────────────────────────┤
│  BobbyPosts.Tokenizer (Pure Elixir)                     │
│  - Uses Bumblebee's tokenizer support                   │
│  - Rust NIFs under the hood (Tokenizers library)        │
├─────────────────────────────────────────────────────────┤
│  BobbyPosts.Qwen3.Model                                 │
│  - 36 transformer layers                                │
│  - Grouped Query Attention (32 Q heads, 8 KV heads)     │
│  - SwiGLU MLP activation                                │
│  - RoPE positional embeddings                           │
├─────────────────────────────────────────────────────────┤
│  BobbyPosts.Qwen3.Attention                             │
│  - quantized_linear_with_lora/4                         │
│  - KV cache for autoregressive generation               │
├─────────────────────────────────────────────────────────┤
│  EMLX.quantized_matmul/7                                │
│  - 4-bit fused matmul (no dequantization overhead)      │
├─────────────────────────────────────────────────────────┤
│  MLX C++ / Metal GPU                                    │
└─────────────────────────────────────────────────────────┘
```

### Phase 4: The LoRA Insight

Initially, we tried "fusing" LoRA adapters into the base model:

```
Base Model + LoRA → Merge → Re-quantize to 4-bit → Fused Model
```

This produced **terrible results**. The outputs were in an "uncanny valley" - too many cat references, weird emoji usage, generic phrasing. The fine-tuning signal was being destroyed.

The fix was obvious in hindsight: **apply LoRA at runtime in full precision**.

```elixir
# attention.ex - The magic formula
def quantized_linear_with_lora(x, base_weights, %{lora_a: a, lora_b: b}, scaling) do
  # Base: 4-bit quantized, fast
  base_output = quantized_linear(x, base_weights)

  # LoRA: fp32, preserves fine-tuning precision
  lora_output = x |> Nx.dot(a) |> Nx.dot(b) |> Nx.multiply(scaling)

  Nx.add(base_output, lora_output)
end
```

The LoRA adapters are tiny (37MB for rank-8 on Qwen3-8B) and stored in fp32. The base model stays quantized (5GB). At inference time:

```
Output = Quantized_Base_Output + (Input @ LoRA_A @ LoRA_B) × Scaling
              ↑                              ↑
         4-bit, fast                   fp32, precise
```

This preserves the full fine-tuning fidelity while keeping memory usage low.

## The Final Stack

### Dependencies

```elixir
# mix.exs
{:nx, "~> 0.10"},
{:emlx, github: "notactuallytreyanastasio/emlx", branch: "feat/quantization-ops"},
{:bumblebee, github: "notactuallytreyanastasio/bumblebee", branch: "feat/qwen3"},
{:safetensors, github: "notactuallytreyanastasio/safetensors_ex"}
```

### Usage

```elixir
# Generate a post
{:ok, [post]} = BobbyPosts.Generator.generate(
  max_tokens: 200,
  temperature: 0.8,
  top_p: 0.9
)

IO.puts(post)
# => "Just tried to make a list of things to do today and it's
#     just a long string of 'maybe' and 'probably not.' But hey,
#     at least I'm not bored."
```

### Performance

| Metric | Value |
|--------|-------|
| Model load time | ~5 seconds |
| Single token latency | ~7ms (135 tok/s) |
| Generation throughput | ~21 tok/s |
| Memory usage | ~5GB (8B model, 4-bit) |

## Key Components

### 1. Safetensors Parser

HuggingFace models use the `.safetensors` format. We built a pure Elixir parser:

```elixir
# Load a specific tensor
{:ok, tensor} = Safetensors.load_tensor(path, header, "model.layers.0.self_attn.q_proj.weight")

# The format is simple:
# [8 bytes: header_size] [header_size bytes: JSON] [tensor data...]
```

### 2. Quantized Weight Loading

MLX 4-bit format stores each weight as three tensors:

```elixir
%{
  weight: tensor,  # uint32 packed int4 values
  scales: tensor,  # bfloat16 scale per group
  biases: tensor   # bfloat16 bias per group
}
```

Group size is typically 64 - every 64 values share one scale/bias pair.

### 3. LoRA Adapter Loading

```elixir
# adapters.safetensors contains pairs like:
# model.layers.20.self_attn.q_proj.lora_a: [4096, 8]
# model.layers.20.self_attn.q_proj.lora_b: [8, 4096]

adapters = AdapterLoader.load_adapters("/path/to/adapters")
# => %{layers: %{20 => %{self_attn: %{q_proj: %{lora_a: ..., lora_b: ...}}}}}
```

### 4. Tokenization

We use Bumblebee's tokenizer support, which wraps the Rust `tokenizers` library:

```elixir
{:ok, tokenizer} = Bumblebee.load_tokenizer({:local, model_path})
tokens = Bumblebee.apply_tokenizer(tokenizer, "Hello world")
text = Bumblebee.Tokenizer.decode(tokenizer, token_ids)
```

## What's Next

This work will be upstreamed:

1. **EMLX quantization ops** → PR to `elixir-nx/emlx`
2. **Safetensors library** → Publish to Hex.pm
3. **Bumblebee Qwen3 docs** → Already in upstream, just docs to add

The goal is for any Elixir developer to run quantized LLMs with:

```elixir
{:emlx, "~> 0.3"},      # With quantization
{:bumblebee, "~> 0.6"}, # Qwen3 support
{:safetensors, "~> 0.1"} # Model loading
```

## Lessons Learned

1. **Runtime LoRA beats fused models** - Don't re-quantize after merging. Apply adapters in full precision at inference.

2. **NIFs are the right abstraction** - Wrapping MLX C++ directly gives zero-overhead GPU access. The Metal kernels do the real work.

3. **Bumblebee's tokenizers are production-ready** - The Rust `tokenizers` library handles all the edge cases. Don't reinvent this.

4. **4-bit quantization is surprisingly good** - For generation tasks, the quality loss is negligible. The memory savings are not.

5. **GenServer is perfect for model serving** - Load once, serve forever. The BEAM's process model handles concurrency naturally.

## Conclusion

We now have a complete, production-ready LLM inference stack in pure Elixir:

- **No Python runtime**
- **No subprocess spawning**
- **Full LoRA fine-tuning fidelity**
- **~21 tok/s on Apple Silicon**

The code is available at [github.com/notactuallytreyanastasio/bobby_posts](https://github.com/notactuallytreyanastasio/bobby_posts).

---

*This blog post was generated by a human, refined by Claude, and will be posted by the very system it describes.*
