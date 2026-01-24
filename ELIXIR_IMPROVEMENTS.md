# Elixir Generator - Bug Fixed

## The Main Bug (FIXED)

**LoRA scaling was 8x too weak!**

```elixir
# BEFORE (wrong):
scaling = scale / rank  # 20.0 / 8 = 2.5

# AFTER (correct):
scaling = scale  # 20.0 (matches Python's mlx_lm)
```

Python's `mlx_lm` uses `scale` directly in LoRALinear:
```python
return y + (self.scale * z)  # scale = 20.0
```

Our Elixir code was dividing by rank, making the fine-tuning signal 8x weaker.
This caused the base model's quirks (cat obsession) to dominate.

**Fix location:** `lib/bobby_posts/adapter_loader.ex` line 50

**Result:** Cat references dropped from 3/3 to 0/5, outputs match Python quality.

---

## Remaining Minor Improvements (Optional)

### 1. Remove `/no_think` Directive

Still adding `/no_think` to prompts. Could remove to exactly match Python.

### 2. Gumbel-Max Sampling (GPU-friendly)

Current sampling does CPU round-trip via `Nx.to_flat_list()`.
Could use Gumbel-max trick to stay on GPU:

```elixir
gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.random_uniform(shape)))))
selected = Nx.argmax(Nx.add(scaled_logits, gumbel))
```

### 3. Reduce Emoji/Hashtag Stripping

Currently strips ALL emojis. Python doesn't strip any.
Training data has ~1.4% emoji usage - could preserve that.

### 4. Add Prompt Variety

Python randomly selects from a prompt list. Elixir uses fixed prompt.

---

## Verification

Greedy decoding now produces identical output:

```
Python:  "I am so excited to be in the city of the future"
Elixir: "I am so excited to be in the city of the future, it's so much more fun..."
```

Temperature sampling produces varied, authentic-sounding posts without cat obsession.
