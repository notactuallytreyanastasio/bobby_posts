# PR Plan: Upstreaming Pure Elixir LoRA Inference Stack

## Overview

Three libraries need work to share this stack with the community:

| Library | Upstream | Status | Action |
|---------|----------|--------|--------|
| EMLX | elixir-nx/emlx | 2 commits ahead | PR quantization ops |
| Bumblebee | elixir-nx/bumblebee | Already upstream! | Minor doc PR |
| Safetensors | None (net-new) | **Publish to Hex.pm** | New package for community |

**Note:** Safetensors is a net-new library that SHOULD exist and be published. There's no Elixir safetensors parser in the HuggingFace ecosystem - this fills that gap.

---

## 1. EMLX Quantization PR

**Branch:** `feat/quantization-ops` on `notactuallytreyanastasio/emlx`
**Target:** `elixir-nx/emlx` main branch

### Changes to PR
- `c_src/emlx_nif.cpp` - 3 new NIFs (63 lines)
  - `quantized_matmul/8` - fused 4-bit matmul
  - `dequantize/6` - unpack quantized weights
  - `quantize/4` - pack weights to 4-bit
- `lib/emlx.ex` - Elixir wrappers (91 lines)
- `test/emlx/quantization_test.exs` - test suite (143 lines)
- `QUANTIZATION.md` - documentation (171 lines)

### Cleanup Before PR
1. **Fix author metadata** - commits show "YOUR_NAME_HERE"
2. **Add docs to mix.exs** - QUANTIZATION.md not in `extras:`
3. **Consider squashing** - 2 commits into 1
4. **Verify tests pass** on clean checkout

### PR Description Template
```markdown
## Add quantization operations for efficient 4-bit inference

This PR adds three operations for working with quantized model weights:

- `quantized_matmul/7` - Fused matrix multiply with 4-bit weights (no dequantization overhead)
- `dequantize/5` - Unpack quantized weights to float
- `quantize/3` - Pack float weights to 4-bit format

### Motivation
Enables efficient LLM inference on Apple Silicon. A quantized Qwen3-8B runs at ~21 tok/s
using 5GB memory (vs 16GB+ for fp16).

### Implementation
Direct wrappers around MLX C++ functions with no Elixir overhead.
```

---

## 2. Bumblebee Documentation PR

**Finding:** Qwen3 model support is already in upstream! (`37953b0`)

**Only delta:** `QWEN3.md` documentation file (193 lines)

### Action
1. Update `bobby_posts/mix.exs` to use upstream Bumblebee (not fork)
2. Optionally PR the QWEN3.md docs to upstream

### bobby_posts Update
```elixir
# Before:
{:bumblebee, github: "notactuallytreyanastasio/bumblebee", branch: "feat/qwen3"}

# After:
{:bumblebee, "~> 0.6"}
```

---

## 3. Safetensors Publication

**Status:** Net-new pure Elixir library - no upstream equivalent exists

**Location:** `notactuallytreyanastasio/safetensors_ex`

### Pre-Publication Cleanup
1. **Add integration tests** with real/synthetic .safetensors files
2. **Remove dead code** in bobby_posts (`lib/bobby_posts/safetensors.ex` is unused copy)
3. **Add README improvements** - link to HuggingFace spec
4. **Version bump** - currently 0.1.0

### Publication Steps
1. Clean up tests
2. Run `mix hex.publish`
3. Update bobby_posts to use hex dependency:
   ```elixir
   {:safetensors, "~> 0.1"}
   ```

---

## 4. bobby_posts Cleanup

After upstream PRs are merged:

### Update Dependencies
```elixir
# mix.exs changes:
{:emlx, "~> 0.3"},  # After quantization PR merged
{:bumblebee, "~> 0.6"},  # Already has Qwen3
{:safetensors, "~> 0.1"},  # After hex publication
```

### Remove Dead Code
- Delete `lib/bobby_posts/safetensors.ex` (226 lines) - uses external dep

### Final Structure
```
bobby_posts/
├── lib/
│   └── bobby_posts/
│       ├── tokenizer.ex      # Pure Elixir (Bumblebee)
│       ├── adapter_loader.ex  # Uses safetensors dep
│       ├── quantized_loader.ex # Uses safetensors dep
│       ├── generator.ex       # Main orchestrator
│       └── qwen3/
│           ├── model.ex       # Transformer forward pass
│           ├── attention.ex   # GQA with LoRA
│           ├── generate.ex    # Autoregressive sampling
│           └── layers.ex      # RMSNorm, RoPE, SwiGLU
```

---

## Execution Order

1. **EMLX PR** (blocking - most value)
   - Clean up author/docs
   - Submit PR to elixir-nx/emlx
   - Work with maintainers on feedback

2. **Safetensors Publish** (can parallel with #1)
   - Add integration tests
   - Publish to hex.pm
   - Update bobby_posts dep

3. **Bumblebee Update** (quick)
   - Switch bobby_posts to upstream Bumblebee
   - Optionally PR QWEN3.md docs

4. **bobby_posts Cleanup** (after #1-3)
   - Update all deps to released versions
   - Remove dead safetensors.ex
   - Tag as v1.1.0

---

## Verification

After all PRs merged and deps updated:
```bash
cd bobby_posts
rm -rf deps _build
mix deps.get
mix compile
mix post 3  # Should generate 3 posts with pure Elixir stack
```

---

## Files to Modify

### EMLX Fork
- `c_src/emlx_nif.cpp` - verify NIF registration
- `lib/emlx.ex` - verify exports
- `mix.exs` - add QUANTIZATION.md to docs
- `.git/config` - verify author settings

### bobby_posts
- `mix.exs` - update all deps
- `lib/bobby_posts/safetensors.ex` - DELETE (uses external dep)

### Safetensors (KEEP AND PUBLISH)
- `test/` - add integration tests
- `README.md` - improve docs
- `mix.exs` - verify hex config
- **Publish to hex.pm as `safetensors`**

---

## 5. Decision Graph Logging (Deciduous)

Track the entire journey in the decision graph:

```bash
# Root goal for the upstreaming work
deciduous add goal "Upstream Pure Elixir LoRA Stack" -c 90 --prompt-stdin << 'EOF'
Share the pure Elixir LoRA inference stack with the community by:
1. PR quantization ops to EMLX
2. Publish safetensors to Hex.pm
3. Update Bumblebee dependency
4. Document the journey
EOF

# Link existing work
deciduous add outcome "v1.0.0 tagged - pure Elixir achieved" -c 100 --commit HEAD
deciduous link <goal_id> <outcome_id> -r "Milestone reached"

# Track each PR
deciduous add action "PR EMLX quantization ops" -c 80
deciduous add action "Publish safetensors to Hex.pm" -c 80
deciduous add action "Write blog post" -c 80
```

---

## 6. Blog Post: "Pure Elixir LLM Inference on Apple Silicon"

### Outline

**Title:** From Python to Pure Elixir: Building a LoRA-Enabled LLM Stack for Apple Silicon

**Sections:**

1. **The Problem**
   - Python ML ecosystem is powerful but heavy
   - Elixir has Nx/Axon but lacks quantized inference
   - Goal: Run fine-tuned LLMs in pure Elixir

2. **The Journey** (reference decision graph)
   - Started with Bumblebee (no quantization support)
   - Discovered EMLX (MLX backend for Nx)
   - Built quantization NIFs
   - Ported safetensors parser
   - Achieved runtime LoRA in Elixir

3. **Architecture Deep Dive**
   ```
   Elixir Code
        ↓
   BobbyPosts.Generator (GenServer)
        ↓
   Qwen3 Model (custom forward pass)
        ↓
   EMLX.quantized_matmul (NIF → MLX C++)
        ↓
   Metal GPU Kernels
   ```

4. **The Key Insight: Runtime LoRA > Fused Model**
   - Fused: merge LoRA → re-quantize → precision loss
   - Runtime: base(4-bit) + LoRA(fp32) = full fidelity

5. **Performance**
   - ~21 tok/s generation
   - ~135 tok/s single-token latency
   - 5GB memory for 8B model

6. **Code Examples**
   - Loading quantized models
   - Applying LoRA adapters
   - Tokenization with Bumblebee

7. **What's Next**
   - Upstream PRs in progress
   - safetensors on Hex.pm
   - Broader model support

### Blog Post Location
Create at: `/Users/robertgrayson/twitter_finetune/bobby_posts/BLOG_POST.md`

---

## 7. Documentation Deliverables

1. **Plan Document** (this file)
2. **Blog Post** - `/bobby_posts/BLOG_POST.md`
3. **EMLX QUANTIZATION.md** - already exists in fork
4. **Decision Graph** - viewable via `deciduous serve`
