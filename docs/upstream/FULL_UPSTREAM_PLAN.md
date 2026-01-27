# Full Upstream Plan: Quantized LLM Inference in Elixir

**Goal:** Enable any Elixir developer to run quantized LLMs with LoRA fine-tuning using standard Bumblebee APIs, with EMLX providing transparent GPU acceleration on Apple Silicon.

## Current State

We have working code across multiple repositories:

| Repository | Purpose | Status |
|------------|---------|--------|
| [bobby_posts](https://github.com/notactuallytreyanastasio/bobby_posts) | Reference implementation | Working |
| [emlx fork](https://github.com/notactuallytreyanastasio/emlx/tree/feat/quantization-ops) | Quantization NIFs | PR #95 closed, feedback received |
| [safetensors_ex](https://github.com/notactuallytreyanastasio/safetensors_ex) | Safetensors parser | Working, not on Hex |
| [bumblebee_quantized](https://github.com/notactuallytreyanastasio/bumblebee_quantized) | Standalone package | Published to GitHub |
| [bobby_posts_adapters](https://github.com/notactuallytreyanastasio/bobby_posts_adapters) | Trained LoRA adapters | Git LFS repo |

## The End Goal

```elixir
# This should just work in any Elixir app
{:ok, model} = Bumblebee.load_model(
  {:hf, "lmstudio-community/Qwen3-8B-MLX-4bit"},
  backend: EMLX.Backend
)

{:ok, adapter} = Bumblebee.load_adapter({:hf, "username/my-adapter"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-8B"})

serving = Bumblebee.Text.generation(model, tokenizer,
  adapter: adapter,
  compile: [batch_size: 1, sequence_length: 256]
)

Nx.Serving.run(serving, "Write a post about Elixir")
```

No custom code. No EMLX-specific calls. Just Bumblebee.

---

## Upstream Strategy

### Stream 1: EMLX (Foundation)

**Maintainer:** Paulo Valente (@polvalente)
**Key Insight:** Don't expose `quantized_matmul` directly. Make `Nx.dot` smart enough to detect quantized tensors and dispatch to the right kernel.

#### Phase 1.1: Quantized Tensor Type
- Add `EMLX.QuantizedTensor` struct
- Implement `EMLX.Backend.from_quantized/4`
- Store quantization metadata (bits, group_size, original_shape)

#### Phase 1.2: Smart Nx.dot Override
- Modify `EMLX.Backend.dot/7` to detect quantized operands
- Dispatch to `quantized_matmul` NIF when appropriate
- Handle transposed cases

#### Phase 1.3: Quantization Utilities
- `EMLX.quantize/3` - float → packed int4
- `EMLX.dequantize/5` - packed int4 → float
- Keep these for debugging/verification

#### Tests Required
```
test/emlx/quantized_tensor_test.exs
test/emlx/quantized_dot_test.exs
test/emlx/quantization_accuracy_test.exs
```

#### PR Strategy
1. Reopen PR #95 with new approach
2. Reference this document
3. Include comprehensive tests
4. Address Paulo's feedback explicitly

---

### Stream 2: Safetensors (Dependency)

**Status:** elixir-nx/safetensors already on Hex (0.1.3)
**Question:** Does it support our use case?

#### Check Existing Package
```elixir
# Does this work?
{:ok, tensors} = Safetensors.read("model.safetensors")

# Can we get raw tensors for quantized weights?
weight = tensors["model.layers.0.self_attn.q_proj.weight"]
scales = tensors["model.layers.0.self_attn.q_proj.scales"]
biases = tensors["model.layers.0.self_attn.q_proj.biases"]
```

#### If Not Sufficient
- Contribute improvements to elixir-nx/safetensors
- OR publish safetensors_ex as separate package

---

### Stream 3: Bumblebee (Integration)

**Maintainers:** José Valim, Jonatan Kłosko
**Dependencies:** EMLX Stream 1 must complete first

#### Phase 3.1: Quantized Model Loading
Add to `Bumblebee.load_model/2`:

```elixir
Bumblebee.load_model({:hf, "model"},
  quantized: true,  # NEW
  backend: EMLX.Backend
)
```

Implementation:
- Detect quantized models from config.json
- Load weight triplets (weight, scales, biases)
- Wrap in `EMLX.QuantizedTensor`
- Store in model params

#### Phase 3.2: Model Definition Compatibility
Existing Axon model definitions should work unchanged because:
- `Nx.dot` handles quantized tensors transparently
- No EMLX-specific code in model definitions

Test with:
- Qwen3
- LLaMA
- Mistral

#### Phase 3.3: LoRA Adapter Support
Add new module `Bumblebee.Adapters`:

```elixir
{:ok, adapter} = Bumblebee.load_adapter({:hf, "user/adapter"})

serving = Bumblebee.Text.generation(model, tokenizer,
  adapter: adapter  # NEW
)
```

Implementation:
- Load LoRA A/B matrices
- Store scaling factor
- Apply at inference: `output + scale * (x @ A @ B)`

---

### Stream 4: Training Integration

**Approach:** Shell out to Python's mlx_lm for training

#### Phase 4.1: Data Preparation
```elixir
Bumblebee.Training.prepare_data(posts, "/path/to/data",
  format: :chatml,
  prompt: "Write a post"
)
```

#### Phase 4.2: Training Wrapper
```elixir
{:ok, adapter} = Bumblebee.Training.lora(
  base_model: {:hf, "Qwen/Qwen3-8B"},
  training_data: "/path/to/data",
  iterations: 25_000,
  rank: 8,
  scale: 20.0
)
```

Implementation:
- Generate mlx_lm config YAML
- Shell out to `python -m mlx_lm.lora`
- Stream training progress
- Return adapter path on completion

---

## Contribution Timeline

### Month 1: EMLX Foundation

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Implement QuantizedTensor | `lib/emlx/quantized_tensor.ex` |
| 2 | Smart Nx.dot override | Modified `lib/emlx/backend.ex` |
| 3 | Comprehensive tests | `test/emlx/quantized_*_test.exs` |
| 4 | PR #95 v2 submission | Reopened PR with new approach |

### Month 2: Bumblebee Loading

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Quantized model detection | Config parsing |
| 2 | Weight triplet loading | Loader modifications |
| 3 | Test with Qwen3 | Integration tests |
| 4 | PR to Bumblebee | Quantized loading PR |

### Month 3: LoRA & Serving

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Adapter loading | `Bumblebee.Adapters` module |
| 2 | Runtime LoRA application | Forward pass modifications |
| 3 | Training integration | `Bumblebee.Training` module |
| 4 | Documentation | Full user guide |

---

## Test Matrix

### EMLX Tests

| Test | What it Validates |
|------|-------------------|
| `quantized_tensor_creation` | QuantizedTensor struct works |
| `dot_with_quantized_right` | Nx.dot detects and dispatches |
| `dot_with_quantized_left` | Transposed case works |
| `quantization_accuracy` | Results match dequantize+dot |
| `mixed_precision` | Float input × int4 weights |

### Bumblebee Tests

| Test | What it Validates |
|------|-------------------|
| `load_quantized_model` | Can load 4-bit model |
| `quantized_inference` | Forward pass produces logits |
| `generation_quality` | Generated text makes sense |
| `adapter_loading` | LoRA adapters load correctly |
| `adapter_application` | LoRA changes outputs |

### Integration Tests

| Test | What it Validates |
|------|-------------------|
| `end_to_end_generation` | Full pipeline works |
| `training_workflow` | Data prep → train → load → generate |
| `performance_benchmark` | Meets speed targets |

---

## Success Criteria

1. **EMLX PR Merged**: Quantization ops in upstream EMLX
2. **Bumblebee PR Merged**: Quantized loading in upstream Bumblebee
3. **Hex Package**: bumblebee_quantized deprecated in favor of core Bumblebee
4. **Documentation**: Full guide on HexDocs
5. **Example App**: Phoenix LiveView demo app

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| EMLX maintainers reject approach | Work closely with Paulo, iterate on design |
| Performance regression | Benchmark at each step, compare to Python |
| Bumblebee API changes | Track Bumblebee releases, update as needed |
| MLX format changes | Pin to specific model versions for testing |

---

## Communication Plan

1. **EMLX**: Comment on PR #95 with this plan
2. **Bumblebee**: Open discussion issue before PRs
3. **ElixirForum**: Post updates for community feedback
4. **Blog Post**: "Quantized LLMs in Elixir" once complete

---

## References

- [EMLX PR #95 Discussion](https://github.com/elixir-nx/emlx/pull/95)
- [bobby_posts README](https://github.com/notactuallytreyanastasio/bobby_posts)
- [MLX Quantization](https://ml-explore.github.io/mlx/build/html/python/quantization.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
