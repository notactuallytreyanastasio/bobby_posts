# Technical Deep Dive: Pure Elixir LLM Inference

A complete inventory of every complex piece required to run a fine-tuned 4-bit quantized LLM in Elixir on Apple Silicon.

---

## Table of Contents

1. [Training Pipeline (Python)](#1-training-pipeline-python)
2. [Quantization Format](#2-quantization-format)
3. [EMLX Fork (C++ NIFs)](#3-emlx-fork-c-nifs)
4. [Safetensors Parser](#4-safetensors-parser)
5. [Qwen3 Architecture Implementation](#5-qwen3-architecture-implementation)
6. [KV Cache](#6-kv-cache)
7. [Runtime LoRA Application](#7-runtime-lora-application)
8. [Tokenization](#8-tokenization)
9. [Sampling & Generation](#9-sampling--generation)
10. [Metal/GPU Pipeline](#10-metalgpu-pipeline)
11. [Phoenix Application Layer](#11-phoenix-application-layer)
12. [Bugs We Found](#12-bugs-we-found)

---

## 1. Training Pipeline (Python)

### What It Does
Fine-tunes Qwen3-8B on custom data using LoRA (Low-Rank Adaptation).

### Components

**Base Model**: `lmstudio-community/Qwen3-8B-MLX-4bit`
- Pre-quantized to 4-bit by LMStudio community
- Weights stored in `.safetensors` format
- ~5GB on disk (vs ~16GB for float16)

**Training Framework**: `mlx_lm.lora`
- Apple's MLX library for training on Apple Silicon
- LoRA: Only trains small adapter matrices, not full weights
- Keeps base model frozen, trains rank-8 adapters

**Training Data Format**: ChatML in JSONL
```json
{"messages": [
  {"role": "user", "content": "write a post"},
  {"role": "assistant", "content": "the actual post content"}
]}
```

**Key Training Parameters**:
```yaml
model: lmstudio-community/Qwen3-8B-MLX-4bit
fine_tune_type: lora
lora_parameters:
  rank: 8        # Low-rank decomposition dimension
  dropout: 0.0   # No dropout during training
  scale: 20.0    # LoRA scaling factor (CRITICAL - see bugs section)
learning_rate: 1e-5
iters: 25000
batch_size: 1
grad_accumulation_steps: 8  # Effective batch size = 8
max_seq_length: 256
mask_prompt: true  # Only train on completions, not prompts
num_layers: 16     # Fine-tune top 16 of 36 layers
```

**Output**: `adapters.safetensors`
- Contains LoRA A and B matrices for each fine-tuned layer
- ~40MB (tiny compared to 5GB base model)

### Command
```bash
mlx_lm.lora --config qwen3_4bit_v4_config.yaml
```

---

## 2. Quantization Format

### The Problem
8B parameters × 2 bytes (float16) = 16GB. Too big for most Macs.

### The Solution: 4-bit Quantization
8B parameters × 0.5 bytes (int4) = 4GB. Fits comfortably.

### How It Works

**Group Quantization**:
- Weights are grouped (typically 64 values per group)
- Each group has its own scale and bias
- `dequantized = (int4_value * scale) + bias`

**Packed Format**:
- 8 int4 values packed into one uint32
- Each int4 is 4 bits, so 8 × 4 = 32 bits = 1 uint32
- Unpacking: `(packed >> (i * 4)) & 0xF`

**Storage Layout** (per linear layer):
```
layer.weight  - uint32[out_features, in_features / 8]  # Packed int4
layer.scales  - bfloat16[out_features, num_groups]     # Scale per group
layer.biases  - bfloat16[out_features, num_groups]     # Bias per group
```

### Why This Matters
Traditional approach:
```
1. Load int4 weights
2. Dequantize to float16 (memory bandwidth bottleneck)
3. Matrix multiply (compute)
```

MLX fused approach:
```
1. Load int4 weights
2. quantized_matmul does dequantize + multiply in one kernel
   (never materializes full float16 weights in memory)
```

---

## 3. EMLX Fork (C++ NIFs)

### Repository
`notactuallytreyanastasio/emlx` (branch: `feat/quantization-ops`)

### What We Added

**File**: `c_src/emlx_nif.cpp`

```cpp
// THE critical operation - fused 4-bit matmul
NIF(quantized_matmul) {
    TENSOR_PARAM(0, x);       // Input: [batch, seq, hidden]
    TENSOR_PARAM(1, w);       // Weights: uint32 packed int4
    TENSOR_PARAM(2, scales);  // Scale factors: bfloat16
    TENSOR_PARAM(3, biases);  // Bias terms: bfloat16
    PARAM(4, bool, transpose);
    PARAM(5, int, group_size);  // Usually 64
    PARAM(6, int, bits);        // 4 for int4

    TENSOR(mlx::core::quantized_matmul(
        *x, *w, *scales, *biases, transpose, group_size, bits, device));
}

// For debugging - expand int4 to float
NIF(dequantize) {
    TENSOR_PARAM(0, w);
    TENSOR_PARAM(1, scales);
    TENSOR_PARAM(2, biases);
    PARAM(3, int, group_size);
    PARAM(4, int, bits);

    TENSOR(mlx::core::dequantize(*w, *scales, *biases, group_size, bits));
}

// Compress float to int4 (for testing)
NIF(quantize) {
    TENSOR_PARAM(0, w);
    PARAM(1, int, group_size);
    PARAM(2, int, bits);

    auto [wq, scales, biases] = mlx::core::quantize(*w, group_size, bits);
    // Return as tuple...
}
```

### Elixir Interface

```elixir
# lib/emlx.ex
def quantized_matmul(x, w, scales, biases, transpose, group_size, bits) do
  EMLX.NIF.quantized_matmul(
    Nx.to_tensor(x),
    Nx.to_tensor(w),
    Nx.to_tensor(scales),
    Nx.to_tensor(biases),
    transpose,
    group_size,
    bits
  )
end
```

### Why NIFs?
- Elixir can't directly call C++ code
- NIFs (Native Implemented Functions) bridge Erlang VM to native code
- EMLX converts Nx tensors ↔ MLX arrays at the boundary
- MLX handles all GPU dispatch internally

---

## 4. Safetensors Parser

### Repository
`notactuallytreyanastasio/safetensors_ex`

### File Format
```
[8 bytes: header_size as little-endian u64]
[header_size bytes: JSON metadata]
[remaining bytes: raw tensor data]
```

### JSON Header
```json
{
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "UI32",
    "shape": [4096, 512],
    "data_offsets": [0, 8388608]
  },
  "model.layers.0.self_attn.q_proj.scales": {
    "dtype": "BF16",
    "shape": [4096, 64],
    "data_offsets": [8388608, 8912896]
  }
}
```

### Elixir Implementation
```elixir
def load(path, backend \\ EMLX.Backend) do
  {:ok, file} = File.open(path, [:read, :binary])

  # Read header size
  <<header_size::little-64>> = IO.binread(file, 8)

  # Parse JSON header
  header_json = IO.binread(file, header_size)
  {:ok, metadata} = Jason.decode(header_json)

  # Memory-map the data section
  data_offset = 8 + header_size
  {:ok, mmap} = :file.open(path, [:read, :binary, :raw])

  # Load each tensor
  for {name, info} <- metadata, into: %{} do
    [start, stop] = info["data_offsets"]
    {:ok, binary} = :file.pread(mmap, data_offset + start, stop - start)

    tensor = binary
      |> Nx.from_binary(dtype_from_string(info["dtype"]))
      |> Nx.reshape(List.to_tuple(info["shape"]))
      |> Nx.backend_transfer(backend)

    {name, tensor}
  end
end
```

### Dtype Mapping
```elixir
def dtype_from_string("F32"), do: :f32
def dtype_from_string("F16"), do: :f16
def dtype_from_string("BF16"), do: :bf16
def dtype_from_string("UI32"), do: :u32  # Packed int4 weights
def dtype_from_string("I32"), do: :s32
```

---

## 5. Qwen3 Architecture Implementation

### Model Structure
```
Qwen3-8B:
├── embed_tokens: [152064, 4096]  # Vocabulary embedding
├── layers: [0..35]               # 36 transformer layers
│   ├── input_layernorm          # RMSNorm
│   ├── self_attn
│   │   ├── q_proj: [4096, 4096]   # Query projection
│   │   ├── k_proj: [4096, 1024]   # Key projection (GQA: 8 KV heads)
│   │   ├── v_proj: [4096, 1024]   # Value projection
│   │   ├── o_proj: [4096, 4096]   # Output projection
│   │   └── q_norm, k_norm         # QK normalization (Qwen3 specific)
│   ├── post_attention_layernorm  # RMSNorm
│   └── mlp
│       ├── gate_proj: [4096, 12288]  # SiLU gate
│       ├── up_proj: [4096, 12288]    # Up projection
│       └── down_proj: [12288, 4096]  # Down projection
├── norm: RMSNorm                 # Final layer norm
└── lm_head: [4096, 152064]       # Output vocabulary projection
```

### Key Architectural Details

**Grouped Query Attention (GQA)**:
- 32 query heads, 8 key/value heads
- Each KV head shared by 4 query heads
- Reduces KV cache size by 4x

**RMSNorm** (not LayerNorm):
```elixir
def rms_norm(x, weight, eps \\ 1.0e-6) do
  variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
  x_norm = Nx.multiply(x, Nx.rsqrt(Nx.add(variance, eps)))
  Nx.multiply(x_norm, weight)
end
```

**SiLU Activation** (not ReLU/GELU):
```elixir
def silu(x) do
  Nx.multiply(x, Nx.sigmoid(x))
end
```

**Rotary Position Embeddings (RoPE)**:
```elixir
def apply_rope(q, k, position_ids, config) do
  {cos, sin} = get_rope_cache(position_ids, config)

  q_rotated = rotate_half(q, cos, sin)
  k_rotated = rotate_half(k, cos, sin)

  {q_rotated, k_rotated}
end

defp rotate_half(x, cos, sin) do
  {x1, x2} = Nx.split(x, 2, axis: -1)
  x_rotated = Nx.concatenate([
    Nx.negate(x2),
    x1
  ], axis: -1)

  Nx.add(
    Nx.multiply(x, cos),
    Nx.multiply(x_rotated, sin)
  )
end
```

### Forward Pass (Simplified)
```elixir
def forward(input_ids, model, kv_cache) do
  # 1. Embedding lookup
  hidden = Nx.take(model.embed_tokens, input_ids)

  # 2. Process through 36 layers
  {hidden, new_cache} =
    Enum.reduce(0..35, {hidden, kv_cache}, fn i, {h, cache} ->
      layer = model.layers[i]

      # Attention block
      residual = h
      h = rms_norm(h, layer.input_layernorm)
      {h, layer_cache} = attention(h, cache[i], layer)
      h = Nx.add(residual, h)

      # MLP block
      residual = h
      h = rms_norm(h, layer.post_attention_layernorm)
      h = mlp(h, layer)
      h = Nx.add(residual, h)

      {h, Map.put(cache, i, layer_cache)}
    end)

  # 3. Final norm + vocabulary projection
  hidden = rms_norm(hidden, model.norm)
  logits = quantized_linear(hidden, model.lm_head)

  {logits, new_cache}
end
```

---

## 6. KV Cache

### Purpose
During autoregressive generation, we generate one token at a time. Without caching, we'd recompute attention over all previous tokens for each new token. KV cache stores the key/value projections from previous positions.

### Structure
```elixir
%{
  0 => %{k: tensor[batch, kv_heads, seq_len, head_dim],
         v: tensor[batch, kv_heads, seq_len, head_dim]},
  1 => %{k: ..., v: ...},
  # ... 36 layers
  35 => %{k: ..., v: ...}
}
```

### Update Logic
```elixir
def update_kv_cache(k_new, v_new, cache, layer_idx) do
  case cache[layer_idx] do
    nil ->
      # First token - initialize cache
      {k_new, v_new, %{k: k_new, v: v_new}}

    %{k: k_cached, v: v_cached} ->
      # Subsequent tokens - concatenate along sequence dimension
      k = Nx.concatenate([k_cached, k_new], axis: 2)
      v = Nx.concatenate([v_cached, v_new], axis: 2)
      {k, v, %{k: k, v: v}}
  end
end
```

### Memory Usage
- Per layer: `2 × batch × 8 heads × seq_len × 128 dim × 2 bytes`
- 36 layers, 256 max seq: ~150MB

---

## 7. Runtime LoRA Application

### What Is LoRA?
Instead of fine-tuning all 8B parameters, train two small matrices A and B:
```
W' = W + (scale × A × B)
where:
  W: [out, in]     # Original weights (frozen)
  A: [out, rank]   # Trained (~40MB total)
  B: [rank, in]    # Trained
  rank: 8          # Much smaller than in/out dimensions
```

### Fused vs Runtime Application

**Fused** (what Python's `mlx_lm` does by default):
```python
# Merge LoRA into weights before inference
W_merged = W + (scale * A @ B)
model.save("merged_model")
```

**Runtime** (what we do in Elixir):
```elixir
# Apply LoRA during each forward pass
def quantized_linear_with_lora(x, layer, lora_layer, scaling) do
  # Base model output (quantized)
  base_out = quantized_matmul(x, layer.weight, layer.scales, layer.biases, ...)

  # LoRA delta (full precision)
  lora_out = x
    |> Nx.dot(lora_layer.lora_a)
    |> Nx.dot(lora_layer.lora_b)
    |> Nx.multiply(scaling)

  Nx.add(base_out, lora_out)
end
```

### Why Runtime?
- Can't easily fuse LoRA into quantized weights (would need requantization)
- Runtime application is mathematically equivalent
- Allows hot-swapping adapters without reloading base model

### Adapter File Structure
```
adapters.safetensors:
├── layers.0.self_attn.q_proj.lora_a: [4096, 8]
├── layers.0.self_attn.q_proj.lora_b: [8, 4096]
├── layers.0.self_attn.v_proj.lora_a: [4096, 8]
├── layers.0.self_attn.v_proj.lora_b: [8, 1024]
├── ... (for each fine-tuned projection)
```

---

## 8. Tokenization

### Tokenizer Source
We use Bumblebee's tokenizer (which wraps HuggingFace tokenizers):

```elixir
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-8B"})
```

### ChatML Format
Qwen3 uses ChatML template:
```
<|im_start|>user
write a post<|im_end|>
<|im_start|>assistant
```

### Special Tokens
```elixir
@bos_token_id 151643      # <|endoftext|>
@eos_token_id 151645      # <|im_end|>
@im_start_id 151644       # <|im_start|>
```

### Encoding Pipeline
```elixir
def encode_prompt(text, tokenizer) do
  prompt = """
  <|im_start|>user
  #{text}<|im_end|>
  <|im_start|>assistant
  """

  Bumblebee.Tokenizer.encode(tokenizer, prompt)
end
```

---

## 9. Sampling & Generation

### Temperature Sampling
```elixir
def sample(logits, temperature \\ 0.7, top_p \\ 0.9) do
  # Scale logits by temperature
  scaled = Nx.divide(logits, temperature)

  # Apply softmax
  probs = Nx.softmax(scaled)

  # Top-p (nucleus) sampling
  sorted_indices = Nx.argsort(probs, direction: :desc)
  sorted_probs = Nx.take(probs, sorted_indices)
  cumsum = Nx.cumsum(sorted_probs)

  # Find cutoff
  cutoff_idx = find_first_above(cumsum, top_p)

  # Zero out tokens below cutoff
  mask = Nx.less(Nx.iota({vocab_size}), cutoff_idx)
  filtered_probs = Nx.select(mask, sorted_probs, 0.0)

  # Renormalize and sample
  normalized = Nx.divide(filtered_probs, Nx.sum(filtered_probs))
  sampled_idx = weighted_random_choice(normalized)

  Nx.take(sorted_indices, sampled_idx)
end
```

### Generation Loop
```elixir
def generate(model, tokenizer, prompt, max_tokens \\ 100) do
  input_ids = encode_prompt(prompt, tokenizer)
  kv_cache = %{}

  {tokens, _cache} =
    Enum.reduce_while(1..max_tokens, {input_ids, kv_cache}, fn _, {ids, cache} ->
      # Forward pass
      {logits, new_cache} = forward(ids, model, cache)

      # Sample next token (only look at last position)
      last_logits = logits[..-1//-1]
      next_token = sample(last_logits)

      # Check for EOS
      if Nx.to_number(next_token) == @eos_token_id do
        {:halt, {ids, new_cache}}
      else
        # Append token, continue with just the new token (KV cache has history)
        {:cont, {next_token, new_cache}}
      end
    end)

  decode(tokens, tokenizer)
end
```

---

## 10. Metal/GPU Pipeline

### The Full Call Stack
```
1. Elixir: Generate.generate_one(model, tokenizer, prompt)
     ↓
2. Elixir: Model.forward(input_ids, model, kv_cache)
     ↓
3. Elixir: Layers.quantized_linear(x, layer)
     ↓
4. Elixir: EMLX.quantized_matmul(x, w, scales, biases, true, 64, 4)
     ↓
5. C++ NIF: emlx_nif.cpp receives Erlang terms
     ↓
6. C++ NIF: Converts to mlx::core::array objects
     ↓
7. MLX C++: mlx::core::quantized_matmul() called
     ↓
8. MLX: Builds lazy computation graph
     ↓
9. MLX: On eval(), compiles graph to Metal shader
     ↓
10. Metal: Dispatches shader to Apple Silicon GPU
      ↓
11. GPU: Executes in parallel across ~1000+ cores
      ↓
12. Metal: Writes result to unified memory
      ↓
13. MLX: Returns mlx::core::array (points to same memory)
      ↓
14. C++ NIF: Wraps as Nx tensor, returns to Elixir
      ↓
15. Elixir: Receives result tensor, continues computation
```

### Why Unified Memory Matters
```
Traditional (NVIDIA):
  CPU RAM ─[PCIe copy]→ GPU VRAM ─[compute]→ GPU VRAM ─[PCIe copy]→ CPU RAM

Apple Silicon:
  Shared RAM ←─[direct access]─→ CPU and GPU
```

No copies needed. The tensor Elixir creates is the same memory the GPU reads/writes.

### MLX Lazy Evaluation
```elixir
# These don't execute immediately - they build a graph
a = Nx.dot(x, w)
b = Nx.add(a, bias)
c = Nx.multiply(b, scale)

# Execution happens when we need the value
result = Nx.to_number(c)  # NOW the graph executes
```

MLX fuses operations where possible, minimizing memory bandwidth.

---

## 11. Phoenix Application Layer

### GenServer for Model State
```elixir
defmodule BobbyPosts.Generator do
  use GenServer

  def start_link(_) do
    GenServer.start_link(__MODULE__, nil, name: __MODULE__)
  end

  def init(_) do
    # Load model on startup (blocking - takes ~5 seconds)
    model = ModelLoader.load_model()
    adapter = AdapterLoader.load_adapter()
    tokenizer = load_tokenizer()

    {:ok, %{model: model, adapter: adapter, tokenizer: tokenizer}}
  end

  def generate(prompt, opts \\ []) do
    GenServer.call(__MODULE__, {:generate, prompt, opts}, :infinity)
  end

  def handle_call({:generate, prompt, opts}, _from, state) do
    result = Generate.generate_one(
      state.model,
      state.adapter,
      state.tokenizer,
      prompt,
      opts
    )
    {:reply, result, state}
  end
end
```

### Mix Task
```elixir
defmodule Mix.Tasks.Post do
  use Mix.Task

  def run(args) do
    Mix.Task.run("app.start")

    count = parse_count(args)

    for _ <- 1..count do
      {:ok, post} = BobbyPosts.Generator.generate("write a post")
      IO.puts(post)
      IO.puts("---")
    end
  end
end
```

### LiveView (Optional Web UI)
```elixir
defmodule BobbyPostsWeb.GeneratorLive do
  use BobbyPostsWeb, :live_view

  def mount(_params, _session, socket) do
    {:ok, assign(socket, posts: [], generating: false)}
  end

  def handle_event("generate", _params, socket) do
    send(self(), :do_generate)
    {:noreply, assign(socket, generating: true)}
  end

  def handle_info(:do_generate, socket) do
    {:ok, post} = BobbyPosts.Generator.generate("write a post")
    {:noreply, assign(socket,
      posts: [post | socket.assigns.posts],
      generating: false
    )}
  end
end
```

---

## 12. Bugs We Found

### Bug 1: LoRA Scaling 8x Too Weak (CRITICAL)

**Symptom**: Outputs had "uncanny valley" quality. Base model quirks dominated.

**Root Cause**:
```elixir
# Our code (WRONG):
scaling = scale / rank  # 20.0 / 8 = 2.5

# Python mlx_lm (CORRECT):
scaling = scale  # 20.0
```

The original LoRA paper uses `alpha / rank`, but `mlx_lm` stores the pre-divided value in `scale`. We were dividing again.

**Impact**: Fine-tuning signal was 8x weaker than intended. The base model's patterns (including cat obsession from Qwen3's training data) dominated outputs.

**Fix**: Change `scaling = scale / rank` to `scaling = scale` in `adapter_loader.ex`.

### Bug 2: Token Escape Character

**Symptom**: Tokenization mismatch between Python and Elixir.

**Root Cause**: Special handling of `\n` in prompt strings.

**Fix**: Ensure prompt strings are processed identically.

### Bug 3: KV Cache Shape Mismatch

**Symptom**: Crash on second token generation.

**Root Cause**: Initial cache shape didn't account for GQA (8 KV heads vs 32 Q heads).

**Fix**: Properly initialize cache with `[batch, 8, seq, 128]` shape.

### Bug 4: RoPE Position Off-by-One

**Symptom**: Slightly degraded output quality, hard to notice.

**Root Cause**: Position IDs started at 1 instead of 0.

**Fix**: Use 0-indexed positions.

### Bug 5: BFloat16 vs Float16 Confusion

**Symptom**: NaN outputs on some layers.

**Root Cause**: Scales/biases are BF16, but we were loading as F16.

**Fix**: Check safetensors dtype field, use `:bf16` for "BF16".

---

## Summary: What You Need to Build This

| Component | Lines of Code | Difficulty | Why It's Hard |
|-----------|--------------|------------|---------------|
| EMLX NIFs | ~300 C++ | High | Undocumented MLX C++ API |
| Safetensors Parser | ~200 Elixir | Medium | Binary format, memory mapping |
| Qwen3 Architecture | ~800 Elixir | High | Many subtle details (GQA, RoPE, etc.) |
| KV Cache | ~100 Elixir | Medium | Shape management across layers |
| Runtime LoRA | ~150 Elixir | Medium | Understanding LoRA math |
| Sampling | ~100 Elixir | Low | Standard algorithms |
| Phoenix App | ~200 Elixir | Low | Standard Phoenix patterns |
| Training Config | ~50 YAML | Low | Once you know the params |

**Total**: ~2000 lines of code across Elixir and C++, plus three forked repositories.

**Key Insight**: The hardest part wasn't any single component - it was understanding how they all connect and getting the subtle details right (like the LoRA scaling bug that took days to find).
