# Bobby Posts

**Pure Elixir inference for a fine-tuned Qwen3-8B-4bit model on Apple Silicon.**

This is what happens when you decide "I want to run my fine-tuned LLM in Elixir" and refuse to give up.

**Status**: Quality now matches Python's `mlx_lm` implementation after fixing a LoRA scaling bug.

## The Journey: Python to Elixir

### It Started Simple

The original goal was straightforward: fine-tune an LLM on my posts and have it generate new ones automatically. Python made this easy:

```python
# The Python version - ~50 lines
from mlx_lm import load, generate
model, tokenizer = load("my-finetuned-model")
post = generate(model, tokenizer, prompt="write a post")
```

**But I wanted it in Elixir.** Not as a Python subprocess. Actually running the model in Elixir, on the GPU.

### The Problem

Bumblebee has Qwen3 support, but not for 4-bit quantized models. Here's what was missing:

1. **No 4-bit quantization support** - Elixir's Nx/EMLX couldn't do quantized matrix multiply
2. **No quantized weight loading** - Bumblebee's loader doesn't handle int4 packed weights
3. **No MLX quantization bindings** - The GPU ops for `quantized_matmul` didn't exist in EMLX
4. **No custom inference path** - Bumblebee's serving doesn't support quantized forward passes

### The Solution: Build Everything

```
What I had to build/fork to make this work:

┌─────────────────────────────────────────────────────────────────┐
│                    THE INSANE STACK                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LAYER 7: Phoenix App (bobby_posts)                             │
│  └─ Web UI, CLI, GenServer to hold model state                  │
│                                                                  │
│  LAYER 6: Qwen3 Quantized Inference (custom implementation)     │
│  └─ model.ex, attention.ex, layers.ex, generate.ex              │
│  └─ Bumblebee has Qwen3, but not quantized inference            │
│  └─ Custom forward pass using quantized_matmul everywhere       │
│  └─ KV cache for autoregressive generation                      │
│                                                                  │
│  LAYER 5: Safetensors Parser (new package)                      │
│  └─ Parse .safetensors file format                              │
│  └─ Load tensors directly into Nx                               │
│  └─ Handle quantized uint32 weight format                       │
│                                                                  │
│  LAYER 4: EMLX Quantization NIFs (fork + new C++ code)          │
│  └─ quantized_matmul - the key operation                        │
│  └─ dequantize, quantize helpers                                │
│  └─ C++ NIFs calling MLX C++ API                                │
│                                                                  │
│  LAYER 3: EMLX (existing, but needed the fork)                  │
│  └─ Nx backend for MLX                                          │
│  └─ Bridges Elixir tensors to MLX arrays                        │
│                                                                  │
│  LAYER 2: MLX (Apple's C++ library)                             │
│  └─ Lazy evaluation, compute graphs                             │
│  └─ Compiles operations to Metal shaders                        │
│  └─ Unified memory management                                   │
│                                                                  │
│  LAYER 1: Metal (Apple's GPU API)                               │
│  └─ Dispatches compute kernels to GPU                           │
│  └─ Manages GPU memory                                          │
│                                                                  │
│  LAYER 0: Apple Silicon                                         │
│  └─ M1/M2/M3/M4 GPU cores                                       │
│  └─ Unified memory (CPU+GPU share RAM)                          │
│  └─ Matrix multiply hardware                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why Each Layer Exists

### Layer 0: Apple Silicon

The hardware that makes this possible. Key insight: **unified memory architecture**.

Traditional GPU setup:
```
CPU RAM ──copy──> GPU VRAM ──compute──> GPU VRAM ──copy──> CPU RAM
         (slow)                                    (slow)
```

Apple Silicon:
```
Shared RAM <──directly accessed by both──> CPU and GPU
                    (no copying!)
```

This means an 8B parameter model that needs ~5GB can just... exist in RAM and be accessed by the GPU. No PCIe bottleneck.

### Layer 1: Metal

Apple's GPU compute API. Like CUDA but for Apple hardware.

MLX compiles operations into Metal shaders - small GPU programs that run in parallel across thousands of cores. A single `matmul` becomes a Metal kernel that tiles the matrices and blasts through the computation.

### Layer 2: MLX

Apple's answer to PyTorch for Apple Silicon. Key features we depend on:

- **Lazy evaluation**: Operations build a graph, only execute when needed
- **Automatic differentiation**: (used for training, not inference)
- **Quantization support**: Native int4 packed format with fused kernels
- **Python AND C++ API**: We use the C++ API from Elixir NIFs

The quantized matmul is crucial. Instead of:
```
dequantize(weights)  # int4 -> float16, memory bandwidth limited
matmul(input, weights)  # float16 x float16
```

MLX does:
```
quantized_matmul(input, packed_weights, scales, biases)
# Unpacks int4, applies scales, multiplies - ALL IN ONE KERNEL
```

### Layer 3: EMLX (Elixir MLX Backend)

Bridges Nx (Elixir's tensor library) to MLX:

```elixir
# When you write this in Elixir:
Nx.tensor([1.0, 2.0, 3.0])

# EMLX converts it to:
mlx::core::array({1.0, 2.0, 3.0})
```

EMLX already existed but was missing quantization ops. That's why we forked it.

### Layer 4: EMLX Quantization NIFs (Our Fork)

**Repository**: [`notactuallytreyanastasio/emlx`](https://github.com/notactuallytreyanastasio/emlx/tree/feat/quantization-ops) (branch: `feat/quantization-ops`)

This is where the actual GPU code lives. We added C++ NIFs to `c_src/emlx_nif.cpp` that expose MLX's quantization functions to Elixir:

```cpp
// c_src/emlx_nif.cpp - the GPU bridge we added (~60 lines of C++)

// quantized_matmul - THE critical operation for 4-bit inference
NIF(quantized_matmul) {
    TENSOR_PARAM(0, x);       // Input tensor from Elixir
    TENSOR_PARAM(1, w);       // Quantized weights (uint32 packed int4)
    TENSOR_PARAM(2, scales);  // Scale factors (bfloat16)
    TENSOR_PARAM(3, biases);  // Bias terms (bfloat16)
    PARAM(4, bool, transpose);
    PARAM(5, int, group_size);
    PARAM(6, int, bits);

    // This single line is where Elixir talks to the GPU:
    // MLX compiles this to a Metal shader that runs on Apple Silicon
    TENSOR(mlx::core::quantized_matmul(
        *x, *w, *scales, *biases, transpose, group_size, bits, device));
}

// Also added: dequantize() and quantize() for debugging/verification
```

**The call chain when you run inference:**
```
Elixir:       EMLX.quantized_matmul(input, weights, scales, biases, true, 64, 4)
                 ↓
C++ NIF:      emlx_nif.cpp receives Erlang terms, converts to MLX arrays
                 ↓
MLX C++:      mlx::core::quantized_matmul() builds compute graph
                 ↓
Metal:        MLX compiles operation to Metal shader (cached after first run)
                 ↓
GPU:          Metal dispatches shader to Apple Silicon GPU cores
                 ↓
Result:       GPU writes output to unified memory, Elixir reads it directly
```

This one function is why we can run 4-bit models efficiently. Without it, we'd have to dequantize every weight tensor on every forward pass (slow, memory-hungry). With it, the GPU unpacks int4→float16, applies scales/biases, and multiplies - all in one fused kernel.

### Layer 5: Safetensors Parser (Our Package)

Model weights are stored in `.safetensors` files. Format is simple:

```
[8 bytes: header size][JSON header][raw tensor bytes...]
```

We wrote a pure Elixir parser that:
1. Reads the header to get tensor names, shapes, dtypes, offsets
2. Memory-maps the file for efficiency
3. Loads tensors directly into Nx with the right backend

For quantized models, weights come in triplets:
- `layer.weight` - uint32 packed int4 values
- `layer.scales` - float16 scale factors
- `layer.biases` - float16 bias terms

### Layer 6: Qwen3 Quantized Inference

Bumblebee already has Qwen3, but its serving layer doesn't support 4-bit quantization. We needed a custom inference implementation that uses `quantized_matmul` for every linear layer. 800+ lines of Elixir:

**Attention (attention.ex)**
```elixir
def forward(hidden_states, attention_mask, kv_cache, model) do
  # Project to Q, K, V
  q = quantized_linear(hidden_states, model.q_proj)
  k = quantized_linear(hidden_states, model.k_proj)
  v = quantized_linear(hidden_states, model.v_proj)

  # Apply rotary position embeddings
  {q, k} = apply_rope(q, k, position)

  # Update KV cache for autoregressive generation
  {k, v, new_cache} = update_kv_cache(k, v, kv_cache)

  # Scaled dot-product attention
  scores = Nx.dot(q, Nx.transpose(k)) / sqrt(head_dim)
  scores = apply_causal_mask(scores, attention_mask)
  attn_weights = Nx.softmax(scores)
  output = Nx.dot(attn_weights, v)

  # Output projection
  quantized_linear(output, model.o_proj)
end
```

**MLP (layers.ex)**
```elixir
def mlp(x, model) do
  # Qwen3 uses SiLU-gated MLP
  gate = quantized_linear(x, model.gate_proj)
  up = quantized_linear(x, model.up_proj)
  x = Nx.multiply(silu(gate), up)
  quantized_linear(x, model.down_proj)
end
```

**Full Forward Pass (model.ex)**
```elixir
def forward(input_ids, model, kv_cache) do
  # Embedding lookup
  hidden = Nx.take(model.embed_tokens, input_ids)

  # 36 transformer layers
  {hidden, new_cache} = Enum.reduce(0..35, {hidden, kv_cache}, fn i, {h, cache} ->
    layer = model.layers[i]

    # Pre-norm architecture
    residual = h
    h = rms_norm(h, layer.input_layernorm)
    {h, layer_cache} = attention(h, mask, cache[i], layer)
    h = Nx.add(residual, h)

    residual = h
    h = rms_norm(h, layer.post_attention_layernorm)
    h = mlp(h, layer)
    h = Nx.add(residual, h)

    {h, put_cache(cache, i, layer_cache)}
  end)

  # Final norm + output projection
  hidden = rms_norm(hidden, model.norm)
  logits = quantized_linear(hidden, model.lm_head)

  {logits, new_cache}
end
```

### Runtime LoRA Application

Unlike fused models where LoRA weights are merged into the base model, we apply LoRA at inference time. This means:

1. **Load base model**: Qwen3-8B-4bit quantized weights
2. **Load adapter**: LoRA A/B matrices from training
3. **Apply at runtime**: For each linear layer, compute `output + (scale × x × A × B)`

```elixir
# adapter_loader.ex - Runtime LoRA application
def apply_lora(output, input, lora_a, lora_b, scaling) do
  # LoRA: W' = W + (scale × A × B)
  # Applied as: y = Wx + scale × (x × A × B)
  lora_out = input
    |> Nx.dot(lora_a)
    |> Nx.dot(lora_b)
    |> Nx.multiply(scaling)

  Nx.add(output, lora_out)
end
```

**Critical Bug We Fixed**: LoRA scaling was 8x too weak.

Python's `mlx_lm` uses `scale` directly (20.0), but our Elixir code was using `scale / rank` (20.0 / 8 = 2.5). This made the fine-tuning signal too weak, causing the base model's quirks to dominate outputs.

```elixir
# BEFORE (wrong):
scaling = scale / rank  # 20.0 / 8 = 2.5

# AFTER (correct - matches Python):
scaling = scale  # 20.0
```

This single-line fix made outputs match Python quality exactly.

### Layer 7: Phoenix Application

Finally, the app that ties it all together:

- **Generator GenServer**: Holds the loaded model in memory
- **Mix Task**: `mix post 5` generates 5 posts
- **LiveView UI**: Web interface for generation
- **Temperature Sampling**: Controls randomness in generation

## The Numbers

What we achieved:

| Metric | Value |
|--------|-------|
| Model | Qwen3-8B-4bit |
| Parameters | 8 billion |
| Memory Usage | ~5GB |
| Model Load Time | 4-6 seconds |
| Single Token Latency | ~7ms (135 tok/s) |
| Generation Throughput | ~21 tok/s |
| Lines of Elixir | ~2000 |
| Lines of C++ (NIFs) | ~300 |

## Training Configuration

The LoRA adapter was trained using `mlx_lm.lora`:

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-8B-MLX-4bit |
| LoRA Rank | 8 |
| LoRA Scale | 20.0 |
| Learning Rate | 1e-5 |
| Iterations | 25,000 |
| Batch Size | 1 (8 gradient accumulation) |
| Max Sequence Length | 256 |
| Layers Fine-tuned | 16 |

Training data: ~1,160 posts in ChatML format with `mask_prompt: true` to only train on completions.

## The Forks

Three repositories we created/forked:

### 1. EMLX Fork
**`notactuallytreyanastasio/emlx`** (branch: `feat/quantization-ops`)

Added:
- `EMLX.quantized_matmul/7`
- `EMLX.dequantize/5`
- `EMLX.quantize/3`

### 2. Safetensors Package
**`notactuallytreyanastasio/safetensors_ex`**

Pure Elixir safetensors parser. Loads model weights into Nx tensors.

### 3. Bumblebee Fork
**`notactuallytreyanastasio/bumblebee`** (branch: `feat/qwen3`)

Used **only for tokenization** (encoding/decoding text to tokens). We don't use Bumblebee's model serving - our custom inference code in bobby_posts handles quantized forward passes directly via EMLX. Bumblebee's Qwen3 tokenizer works great, but its serving layer doesn't support 4-bit quantization.

### 4. LoRA Adapters Repository
**[`notactuallytreyanastasio/bobby_posts_adapters`](https://github.com/notactuallytreyanastasio/bobby_posts_adapters)**

Git LFS repository containing the trained LoRA adapter weights (~38MB each). Versioned separately from code.

```bash
# Clone with Git LFS
git lfs install
git clone https://github.com/notactuallytreyanastasio/bobby_posts_adapters
```

Current version: **v5** (trained on longer posts >160 chars, Bluesky weighted 3x)

## Usage

```bash
# Generate posts
mix post           # 1 post
mix post 5         # 5 posts
mix post "prompt"  # with custom prompt

# Web UI
mix phx.server
# http://localhost:4000
```

## What Made This Hard

1. **No documentation** for MLX C++ API - had to read source code
2. **Quantization format** - figuring out the int4 packing took days
3. **KV cache** - getting the shapes right for autoregressive generation
4. **Debugging GPU code** - errors are cryptic, no stack traces
5. **Token mismatch bug** - spent hours finding an escape character issue
6. **LoRA scaling bug** - outputs had an "uncanny valley" quality until we discovered Python uses `scale` directly, not `scale/rank` like the original LoRA paper

## Was It Worth It?

The Python version is 50 lines. This is 2000+ lines of Elixir plus C++ NIFs plus three forked packages.

But now I have:
- LLM inference in my native language (Elixir)
- No Python runtime dependency for inference
- Full control over the generation loop
- A GenServer holding a GPU-accelerated model
- The satisfaction of doing something arguably insane

## License

MIT
