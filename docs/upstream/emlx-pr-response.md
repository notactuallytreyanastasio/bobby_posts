# EMLX PR #95 Response Draft

**PR:** https://github.com/elixir-nx/emlx/pull/95
**Status:** CLOSED (not merged)
**Maintainer Feedback:** Paulo Valente asked for use case explanation and architectural discussion

---

## Response to Maintainer Feedback

@polvalente Thanks for the feedback! I accidentally closed it while testing something. I'd love to reopen and discuss.

## Use Case

I'm running 4-bit quantized LLMs (Qwen3-8B-4bit) in Elixir for local inference on Apple Silicon. The full project: https://github.com/notactuallytreyanastasio/bobby_posts

Without these ops, there's no way to run quantized models efficiently in EMLX - you'd need to dequantize every weight tensor on every forward pass (slow, 4x memory).

**Performance achieved:**
- Model: Qwen3-8B-4bit (~5GB memory)
- Single token latency: ~7ms (135 tok/s)
- Generation throughput: ~21 tok/s
- Hardware: Apple Silicon with unified memory

## Why a Separate Function vs Nx.dot Override

The quantized matmul has a fundamentally different signature:

```elixir
# Normal matmul
Nx.dot(input, weights)  # 2 tensors

# Quantized matmul
EMLX.quantized_matmul(input, weights, scales, biases, transpose, group_size, bits)
# weights is uint32 packed int4, scales/biases are bf16 per-group
```

The quantized version needs 4 tensors + 3 params because:
1. `weights` - packed uint32 (8 int4 values per uint32)
2. `scales` - per-group scale factors (64 weights share 1 scale)
3. `biases` - per-group zero points

There's no way to express this through `Nx.dot` without significant API changes.

## Alternative: Nx.LinAlg.quantized_matmul?

One option could be adding this to Nx itself as `Nx.LinAlg.quantized_matmul/7` with backend dispatch. Then EMLX (and potentially other backends) could implement it. But that's a bigger scope change.

For now, exposing it as `EMLX.quantized_matmul` lets people who need quantized inference use it today while we figure out the right Nx-level API.

## What the PR Adds

**Files changed:**
- `c_src/emlx_nif.cpp` - 63 lines of C++ NIF code
- `lib/emlx.ex` - 91 lines of Elixir API
- `test/emlx/quantization_test.exs` - 143 lines of tests

**Functions:**
- `EMLX.quantized_matmul/7` - fused 4-bit matmul (the key operation)
- `EMLX.dequantize/5` - expand int4 to float (for debugging)
- `EMLX.quantize/3` - compress float to int4

## Next Steps

1. Reopen PR #95
2. Post this response as a comment
3. Discuss architectural approach with maintainers
4. Iterate based on feedback
