# I Wrote 2000 Lines of Code So A Computer Could Post Like Me While I Sleep

So I had this normal, healthy thought: what if I trained an AI on all my tweets and had it post to Bluesky automatically? You know, as one does. The kind of project that definitely justifies the time investment and isn't at all a cry for help.

The Python version was 50 lines. It worked. It posted. I could have stopped there.

I did not stop there.

## The Part Where Everything Was Fine

Training was actually pretty straightforward. Export 42,418 tweets (I have posted too much), format them for fine-tuning, run LoRA on Qwen3-8B. The model learned to sound like me, which is either a testament to machine learning or a damning indictment of how predictable I am.

The validation loss went from 7.0 to 3.5. I don't know if those numbers are good. They went down. That seems correct.

```python
from mlx_lm import load, generate
model, tokenizer = load("Qwen3-8B-MLX-4bit", adapter_path="adapters")
response = generate(model, tokenizer, prompt="Write a tweet...")
```

Fifty lines. Works on my Mac. Posts to Bluesky. Twenty-one tokens per second. This is the reasonable stopping point that a normal person would recognize and accept.

## The Part Where I Made A Series of Decisions

The Python bot spawned a new process every time it generated a post. Each process loaded 5GB of model weights from scratch. This took like 5 seconds of startup before generating even one token.

Now look. Is 5 seconds a long time? Not really. Is it a problem that matters for a bot that posts maybe once an hour? Absolutely not. Did I decide this was unacceptable and that I needed to rewrite everything in Elixir so a GenServer could hold the model in memory permanently?

Obviously yes.

The rational person would ask "why Elixir specifically" and the answer is that I already had an Elixir project open and I didn't feel like closing it. Architecture is mostly about what tabs you have open.

## The Part Where The Obvious Thing Didn't Work

So my first thought was to use Bumblebee with EXLA, which is the normal Elixir ML stack. I spent an hour setting this up before discovering that EXLA doesn't support Apple's Metal GPU. It only works with CUDA (nvidia), ROCm (AMD), and TPUs (google's thing that you don't have).

This is documented. I did not read the documentation. Classic.

So now I can't use the standard approach and I've already told myself this project is happening, which means I need to find another way to run GPU inference on an M-series Mac from Elixir. The reasonable thing would be to accept defeat and go back to Python. Instead I found EMLX.

## EMLX: A Library That Existed And Was Almost What I Needed

EMLX wraps Apple's MLX framework for Elixir. It has 118 different NIFs (native implemented functions - basically Elixir calling C++). This seemed perfect until I realized it was missing the three specific operations I needed for 4-bit quantized inference.

4-bit quantization is the thing that lets you run an 8-billion-parameter model in 5GB instead of 16GB. Without it, the model literally doesn't fit on my laptop. So this wasn't a "nice to have" situation.

The normal response here is "ah well, this library doesn't support my use case, guess I'll wait for someone to add it or use something else."

I forked the library and added 300 lines of C++ instead.

This is what happens when you don't have enough hobbies.

## Writing C++ NIFs Like A Person Who Knows What They're Doing

I did not know what I was doing.

MLX has a C++ API. EMLX already had examples of wrapping MLX functions. I mostly just copied the patterns and changed the function names. The three NIFs I needed were:

- `quantized_matmul` - multiplies matrices using packed 4-bit weights
- `dequantize` - unpacks 4-bit to regular floats
- `quantize` - packs floats to 4-bit

```cpp
{"quantized_matmul", 8, quantized_matmul},
{"dequantize", 6, dequantize},
{"quantize", 4, quantize}
```

At runtime, MLX compiles these to Metal shaders. I don't really understand how that works but the numbers go in and different numbers come out and they match what Python produces, so I'm calling it a win.

## Building The Rest of the Stack

With quantization working I could actually load the model. This required:

1. A safetensors parser because HuggingFace uses a format that no Elixir library can read (I checked) (there are no Elixir libraries for this) (I had to write one)

2. A quantized weight loader that understands how MLX packs int4 values into uint32s with separate scale/bias tensors

3. An entire Qwen3 transformer implementation - 36 layers of grouped query attention with RoPE embeddings and SwiGLU MLPs

4. A generation module with KV caching for autoregressive decoding

5. A tokenizer wrapper (thankfully Bumblebee's tokenizers actually work, small mercies)

6. A GenServer to hold all this state and expose a nice API

This is roughly 2000 lines of Elixir. The Python version was 50 lines. I am very normal.

The good news is that single-token inference hit 135 tokens per second, which is 14x faster than Python. The bad news is that nobody cares about single-token latency for a bot that posts once an hour. The generation speed is the same because the actual matrix multiplication happens on the GPU either way.

I optimized the wrong thing. Very on brand.

## The Part Where The Output Was Haunted

So I got everything working and ran generation and the output was... wrong.

Not wrong like "error" wrong. Wrong like "uncanny valley" wrong. The bot was posting about cats constantly. Using emojis I never use. The phrasing was generically AI-ish instead of specifically me-ish.

I spent hours debugging. The tensor shapes were correct. The attention masks looked right. Single-token forward passes matched Python exactly. But multi-token generation produced cursed output.

Eventually I figured it out: I was using a "fused" model where the LoRA adapters had been merged into the base weights and then re-quantized to 4-bit. This is a common deployment pattern. It is also wrong.

When you re-quantize after merging, you snap every weight to the nearest 4-bit value. The fine-tuning adjustments - the thing that made the model sound like me - get rounded away. The aggregate effect is the model forgetting what it learned.

The Python library doesn't do this. It keeps the base model quantized and applies LoRA adapters at runtime in full precision:

```
Output = QuantizedBaseModel(input) + (input @ LoRA_A @ LoRA_B) * scale
```

Base model stays 4-bit (5GB, fast). Adapters stay fp32 (37MB, precise). Add them together at inference time. The math is identical but you never lose precision from re-quantization.

I fixed this in like 20 lines of Elixir once I understood the problem. The previous 2 days of debugging were "character building."

## The Part Where It Actually Worked

With runtime LoRA applied, the output matched Python exactly. The bot started posting things that actually sounded like me. Mission accomplished.

```elixir
{:ok, [post]} = BobbyPosts.Generator.generate(max_tokens: 200, temperature: 0.8)
```

One line to generate. The GenServer holds the model. No Python subprocess. No 5-second cold starts. Just inference on demand.

I replaced 50 lines of Python with 2000+ lines of Elixir, 300 lines of C++, and three new packages that I now have to maintain forever. The generation speed is identical. The memory usage is identical. The only measurable improvement is eliminating startup overhead that didn't matter.

This is what winning looks like.

## Things I Learned

**EXLA doesn't do Metal.** I should have read the docs. You should also read the docs. Neither of us will actually do this.

**Re-quantizing kills fine-tuning.** If you merge LoRA and re-quantize, you lose the signal. Apply adapters at runtime instead. This is documented in exactly zero places that I could find.

**NIFs are fine actually.** Everyone acts like native code is scary but it's really just "call a C function with these arguments." The scary part is when you mess up memory management but MLX handles all that.

**The BEAM is good at holding state.** GenServers are basically "load a thing once and let everyone use it forever." Perfect for ML models that take seconds to load but milliseconds to run.

**I am not immune to scope creep.** I knew this already.

## The Stack

In case anyone wants to do this for some reason, here's what I ended up with:

```
Phoenix LiveView (web UI)
       ↓
Generator GenServer (holds model in memory)
       ↓
Qwen3 Transformer (36 layers, GQA, RoPE, SwiGLU)
       ↓
Quantized Weight Loader + Safetensors Parser
       ↓
EMLX Quantization NIFs (quantized_matmul, dequantize, quantize)
       ↓
MLX C++ (lazy eval, Metal codegen)
       ↓
Metal GPU (Apple's shader language thing)
       ↓
Some silicon that Apple put in my laptop
```

Seven layers of abstraction between "generate a post" and "GPU goes brrr." Totally reasonable.

## Upstream Work

I'm planning to PR the quantization NIFs back to the main EMLX repo and publish the safetensors parser to Hex. This way other people can make questionable decisions without having to write their own C++.

You're welcome, future me.

## Conclusion

I built a 7-layer GPU inference stack in Elixir so that a GenServer could generate tweets while I sleep. The output is indistinguishable from what Python produces. The engineering investment was wildly disproportionate to the problem being solved.

The bot is live at [@bobbby.online](https://bsky.app/profile/bobbby.online). Sometimes it posts about cats because the fine-tuning isn't perfect. Sometimes it posts bangers. Either way, I'm not the one doing it anymore.

All 110 decisions are tracked in a decision graph if you want to see exactly how this spiraled out of control: [view the graph](https://notactuallytreyanastasio.github.io/bobby_posts/)

Writing documentation is single-handedly the best way to realize your idea was terrible.

---

*total lines of code: ~2300*
*total lines actually needed: ~50*
*ratio: concerning*
