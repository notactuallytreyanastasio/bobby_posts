# Safetensors Hex Package Plan

**Repo:** https://github.com/notactuallytreyanastasio/safetensors_ex
**Goal:** Publish to Hex.pm as `safetensors`

---

## What It Does

Pure Elixir parser for HuggingFace `.safetensors` format:
- Parse header (tensor names, shapes, dtypes, offsets)
- Load tensors directly into Nx
- Handle quantized dtypes (uint32 packed int4)

## Current State

Working code in `lib/bobby_posts/safetensors.ex` - needs extraction to standalone package.

## Before Publishing

1. [ ] Extract to standalone mix project
2. [ ] Add comprehensive tests
3. [ ] Add @moduledoc and @doc
4. [ ] Add typespec coverage
5. [ ] Write README with examples
6. [ ] Check for existing `safetensors` package on Hex
7. [ ] Publish to Hex.pm

## Example Usage

```elixir
# Read header
{:ok, {header, data_offset}} = Safetensors.read_header("model.safetensors")

# Load specific tensor
{:ok, tensor} = Safetensors.read_tensor(path, header, "model.layers.0.self_attn.q_proj.weight")

# Load all tensors
{:ok, tensors} = Safetensors.load("model.safetensors", backend: {EMLX.Backend, device: :gpu})
```

## File Format

```
[8 bytes: header size as u64 little-endian]
[N bytes: JSON header with tensor metadata]
[remaining bytes: raw tensor data]
```

Each tensor entry in header:
```json
{
  "tensor_name": {
    "dtype": "BF16",
    "shape": [4096, 4096],
    "data_offsets": [0, 33554432]
  }
}
```
