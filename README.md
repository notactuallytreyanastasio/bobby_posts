# Bobby Posts

**Pure Elixir inference for a fine-tuned Qwen3-8B-4bit model on Apple Silicon.**

This Phoenix application generates social media posts in a specific voice using a locally-running 4-bit quantized LLM. It achieves ~135 tok/s single-token latency and ~21 tok/s generation throughput on M-series Macs.

## The Big Picture

This is the culmination of a project to run fine-tuned LLM inference entirely in Elixir on Apple Silicon. The pipeline:

1. **Training**: Fine-tune Qwen3-8B on personal posts using MLX + LoRA (Python)
2. **Model Fusion**: Merge LoRA adapters into base model, quantize to 4-bit
3. **Inference**: Pure Elixir inference using custom EMLX quantization NIFs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        bobby_posts                               │
│                    (Phoenix Application)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Web UI     │    │   Mix Task   │    │  Generator   │       │
│  │  (LiveView)  │    │  (mix post)  │    │  (GenServer) │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┴───────────────────┘                │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │   Qwen3 Model   │                          │
│                    │  (Pure Elixir)  │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
├─────────────────────────────┼────────────────────────────────────┤
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │                    Dependencies                          │    │
│  │                                                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │    EMLX     │  │ Safetensors │  │  Bumblebee  │      │    │
│  │  │ (Quant NIFs)│  │  (Weights)  │  │  (Qwen3)    │      │    │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────┘      │    │
│  │         │                │                               │    │
│  │         └────────┬───────┘                               │    │
│  │                  │                                       │    │
│  │         ┌────────▼────────┐                              │    │
│  │         │   MLX (Metal)   │                              │    │
│  │         │  Apple Silicon  │                              │    │
│  │         └─────────────────┘                              │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Dependencies (Forks)

This project uses three forked Elixir packages:

### 1. EMLX (feat/quantization-ops branch)
**Repository**: `notactuallytreyanastasio/emlx`

Added quantization NIFs to enable 4-bit model inference:
- `EMLX.quantized_matmul/7` - Matrix multiply with packed int4 weights
- `EMLX.dequantize/5` - Convert quantized weights to float
- `EMLX.quantize/3` - Quantize float tensor to packed format

### 2. Safetensors
**Repository**: `notactuallytreyanastasio/safetensors_ex`

Pure Elixir safetensors file parser for loading model weights directly into Nx tensors.

### 3. Bumblebee (feat/qwen3 branch)
**Repository**: `notactuallytreyanastasio/bumblebee`

Fork tracking upstream Bumblebee. Qwen3 support already exists in upstream - this fork is used for compatibility with our custom quantized inference pipeline (Bumblebee's standard serving doesn't support 4-bit quantization).

## Installation

```bash
git clone https://github.com/notactuallytreyanastasio/bobby_posts
cd bobby_posts
mix deps.get
mix compile
```

## Usage

### CLI
```bash
# Generate a single post
mix post

# Generate multiple posts
mix post --count 5

# Custom prompt
mix post --prompt "write about coffee"
```

### Web UI
```bash
mix phx.server
# Visit http://localhost:4000
```

### Programmatic
```elixir
# Generate posts
{:ok, posts} = BobbyPosts.Generator.generate(count: 3)

# With options
{:ok, posts} = BobbyPosts.Generator.generate(
  prompt: "write about debugging",
  max_tokens: 100,
  temperature: 0.8,
  top_p: 0.95
)
```

## Configuration

The model path defaults to `/Users/robertgrayson/twitter_finetune/fused_model`. Override in config:

```elixir
config :bobby_posts, :model_path, "/path/to/your/model"
```

## Generation Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `max_tokens` | 100 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling temperature (0 = greedy) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `char_limit` | 300 | Bluesky character limit |

## How It Works

### GPU Acceleration Stack

The entire inference runs on Apple Silicon GPU via this stack:

```
┌─────────────────────────────────────────────────────────────┐
│                     Elixir Application                       │
│                   (bobby_posts, Nx tensors)                  │
├─────────────────────────────────────────────────────────────┤
│                         EMLX                                 │
│              (Nx Backend + Quantization NIFs)                │
│                                                              │
│  Nx.tensor([1,2,3]) -> EMLX.Backend -> mlx::array           │
│  EMLX.quantized_matmul() -> C++ NIF -> mlx::quantized_matmul│
├─────────────────────────────────────────────────────────────┤
│                     MLX (C++ Library)                        │
│            Apple's ML framework for Apple Silicon            │
│                                                              │
│  - Lazy evaluation (builds compute graph)                    │
│  - Unified memory (CPU/GPU share RAM)                        │
│  - Metal shader compilation                                  │
├─────────────────────────────────────────────────────────────┤
│                     Metal Framework                          │
│              Apple's GPU compute API                         │
│                                                              │
│  - GPU kernel dispatch                                       │
│  - Memory management                                         │
│  - Shader execution                                          │
├─────────────────────────────────────────────────────────────┤
│                   Apple Silicon GPU                          │
│              (M1/M2/M3/M4 Neural Engine)                     │
│                                                              │
│  - Matrix multiply units                                     │
│  - Unified memory architecture (no CPU<->GPU copies)         │
│  - 16GB+ shared RAM for large models                         │
└─────────────────────────────────────────────────────────────┘
```

### Why This Is Fast

1. **Unified Memory**: Apple Silicon shares RAM between CPU and GPU. Tensors created in Elixir are directly accessible by the GPU - no copying.

2. **Quantized Matmul**: The `EMLX.quantized_matmul` NIF calls MLX's native quantized matrix multiply. The GPU unpacks int4 values and multiplies in a single fused kernel - no separate dequantization pass.

3. **Lazy Evaluation**: MLX builds a compute graph and only executes when results are needed. This enables kernel fusion and optimal scheduling.

4. **Metal Shaders**: MLX compiles optimized Metal shaders for each operation. These run directly on Apple Silicon's GPU cores.

### Token Generation Loop

1. **Tokenization**: Text -> tokens via Python subprocess (HuggingFace tokenizers)
2. **Prefill**: Process all prompt tokens, build KV cache
3. **Generation**: Autoregressive token generation with temperature sampling
4. **Decoding**: Tokens -> text via Python subprocess

### Quantized Inference

The model uses MLX's 4-bit quantization format:
- 8 int4 values packed per uint32
- Group size of 64 for scales/biases
- `quantized_matmul` NIF bypasses dequantization for speed

```elixir
# What happens under the hood:
input = Nx.tensor([[1.0, 2.0, ...]])           # Elixir tensor
w = load_quantized_weights()                    # uint32 packed int4
scales = load_scales()                          # float16
biases = load_biases()                          # float16

# This single call:
output = EMLX.quantized_matmul(input, w, scales, biases, true, 64, 4)

# Compiles to Metal shader that:
# 1. Unpacks 8 int4 values from each uint32
# 2. Applies group-wise scales and biases
# 3. Multiplies with input
# 4. All in one GPU kernel, no intermediate buffers
```

### KV Cache

Key-value cache enables efficient autoregressive generation:
- Cache grows with sequence length
- Single-token forward passes after prefill
- Enables 135 tok/s latency

## Performance

On Apple M-series:
- **Model loading**: ~4-6 seconds
- **Prefill**: ~150ms for typical prompts
- **Generation**: ~21 tok/s sustained
- **Memory**: ~5GB for 8B-4bit model

## Project Structure

```
bobby_posts/
├── lib/
│   ├── bobby_posts/
│   │   ├── generator.ex          # GenServer holding model
│   │   ├── quantized_loader.ex   # Safetensors -> Nx tensors
│   │   ├── safetensors.ex        # File format parser
│   │   └── qwen3/
│   │       ├── model.ex          # Forward pass
│   │       ├── attention.ex      # Multi-head attention
│   │       ├── layers.ex         # RMSNorm, MLP, etc.
│   │       └── generate.ex       # Token generation loop
│   ├── bobby_posts_web/
│   │   └── live/
│   │       └── generate_live.ex  # LiveView UI
│   └── mix/tasks/
│       └── post.ex               # CLI task
└── mix.exs
```

## Related Projects

- **bluesky_bot.py** - Python version using MLX-LM (for comparison)
- **elixir_qwen3/** - Earlier pure-Elixir experiment
- **fused_model/** - The actual model weights (not in repo)

## License

MIT
