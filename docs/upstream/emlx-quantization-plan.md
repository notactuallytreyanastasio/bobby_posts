# EMLX Quantization Upstream Plan

**Goal:** Enable transparent quantized inference through standard Nx operations, allowing Bumblebee models to work with quantized weights without EMLX-specific code.

## Maintainer Feedback (Paulo Valente)

From [PR #95](https://github.com/elixir-nx/emlx/pull/95):

> "If you do wanna explore something along the lines of what I think is gonna be needed, you can try adding a backend option for tensors in EMLX that will annotate whether the tensor is, for instance, s4 quantized."
>
> "And then operations such as Nx.dot can branch based on that to invoke the proper quantized matmul implementations."
>
> "This would allow you to possibly even upstream the model to Bumblebee and we could use EMLX as a pure backend instead of having to drop down to lower levels of abstraction with calling EMLX directly"

## The Vision

### Current Approach (EMLX-specific)
```elixir
# Must call EMLX directly - can't use standard Nx
result = EMLX.quantized_matmul(x, weight, scales, biases, true, 64, 4)
```

### Proposed Approach (Transparent via Nx)
```elixir
# Load quantized tensor with metadata
weight = EMLX.from_quantized(packed_weight, scales, biases,
  bits: 4,
  group_size: 64
)

# Standard Nx.dot automatically uses quantized_matmul
result = Nx.dot(x, weight)
```

## Implementation Plan

### Phase 1: Quantized Tensor Type in EMLX

Add a new tensor representation that carries quantization metadata:

```elixir
defmodule EMLX.QuantizedTensor do
  @moduledoc """
  A tensor that stores quantized weights with their scales and biases.

  When used in Nx operations, EMLX automatically invokes the
  appropriate quantized kernel.
  """

  defstruct [:weight, :scales, :biases, :bits, :group_size, :original_shape]

  @type t :: %__MODULE__{
    weight: Nx.Tensor.t(),      # Packed uint32 (8 x int4 per element)
    scales: Nx.Tensor.t(),      # Per-group scale factors (bf16)
    biases: Nx.Tensor.t(),      # Per-group zero points (bf16)
    bits: 4 | 8,                # Quantization bits
    group_size: pos_integer(),  # Weights per scale/bias group
    original_shape: tuple()     # Shape before quantization
  }
end
```

### Phase 2: Backend Tensor Annotation

Extend EMLX.Backend to recognize quantized tensors:

```elixir
defmodule EMLX.Backend do
  # Existing backend code...

  @doc """
  Creates a quantized tensor from packed weights and scales/biases.
  """
  def from_quantized(weight, scales, biases, opts \\ []) do
    bits = Keyword.get(opts, :bits, 4)
    group_size = Keyword.get(opts, :group_size, 64)

    # Store as a special tensor with metadata
    # The metadata travels with the tensor through operations
    %EMLX.QuantizedTensor{
      weight: weight,
      scales: scales,
      biases: biases,
      bits: bits,
      group_size: group_size,
      original_shape: compute_original_shape(weight, bits)
    }
  end

  @doc """
  Checks if a tensor is quantized.
  """
  def quantized?(%EMLX.QuantizedTensor{}), do: true
  def quantized?(_), do: false
end
```

### Phase 3: Override Nx.dot in Backend

Modify EMLX's `dot` implementation to detect quantized tensors:

```elixir
defmodule EMLX.Backend do
  # In the Nx.Backend behaviour implementation

  @impl true
  def dot(out, left, contract_left, batch_left, right, contract_right, batch_right) do
    cond do
      # If right operand is quantized, use quantized_matmul
      quantized?(right) ->
        quantized_dot(out, left, right, contract_left, contract_right)

      # If left operand is quantized (transposed case)
      quantized?(left) ->
        quantized_dot_transposed(out, left, right, contract_left, contract_right)

      # Standard dot
      true ->
        standard_dot(out, left, contract_left, batch_left, right, contract_right, batch_right)
    end
  end

  defp quantized_dot(out, input, %EMLX.QuantizedTensor{} = qweight, _contract_left, _contract_right) do
    # Unpack quantized tensor
    %{weight: w, scales: s, biases: b, group_size: gs, bits: bits} = qweight

    # Call quantized_matmul NIF
    EMLX.NIF.quantized_matmul(
      to_ref(input),
      to_ref(w),
      to_ref(s),
      to_ref(b),
      true,  # transpose
      gs,
      bits,
      device(out)
    )
    |> wrap_tensor(out)
  end
end
```

### Phase 4: Safetensors Integration

Update the loader to create quantized tensors:

```elixir
defmodule EMLX.Safetensors do
  @doc """
  Loads a quantized weight from safetensors format.

  Expects weight triplets: {name}.weight, {name}.scales, {name}.biases
  """
  def load_quantized(path, header, base_name, opts \\ []) do
    weight = load_tensor(path, header, "#{base_name}.weight")
    scales = load_tensor(path, header, "#{base_name}.scales")
    biases = load_tensor(path, header, "#{base_name}.biases")

    EMLX.Backend.from_quantized(weight, scales, biases, opts)
  end
end
```

### Phase 5: Bumblebee Integration

With transparent quantized tensors, Bumblebee models work unchanged:

```elixir
# In Bumblebee's Qwen3 model - NO CHANGES NEEDED
defn attention(hidden, params, opts) do
  # This just works - Nx.dot detects quantized params.q_proj
  q = Nx.dot(hidden, params.q_proj)
  k = Nx.dot(hidden, params.k_proj)
  v = Nx.dot(hidden, params.v_proj)
  # ...
end
```

The only change is in the loader:

```elixir
# Bumblebee loader addition
defp load_params(path, opts) do
  if opts[:quantized] do
    # Load as quantized tensors
    EMLX.Safetensors.load_quantized(path, header, name)
  else
    # Standard loading
    Safetensors.load_tensor(path, header, name)
  end
end
```

## Test Plan

### Unit Tests for EMLX

```elixir
defmodule EMLX.QuantizedTest do
  use ExUnit.Case, async: true

  describe "from_quantized/4" do
    test "creates quantized tensor with correct metadata" do
      # Create a small quantized tensor
      weight = Nx.tensor([[1, 2], [3, 4]], type: :u32)
      scales = Nx.tensor([[1.0], [1.0]], type: :bf16)
      biases = Nx.tensor([[0.0], [0.0]], type: :bf16)

      qt = EMLX.Backend.from_quantized(weight, scales, biases, bits: 4, group_size: 64)

      assert %EMLX.QuantizedTensor{} = qt
      assert qt.bits == 4
      assert qt.group_size == 64
    end

    test "quantized?/1 returns true for quantized tensors" do
      qt = create_quantized_tensor()
      assert EMLX.Backend.quantized?(qt)
    end

    test "quantized?/1 returns false for regular tensors" do
      t = Nx.tensor([1, 2, 3])
      refute EMLX.Backend.quantized?(t)
    end
  end

  describe "Nx.dot with quantized tensors" do
    test "automatically uses quantized_matmul" do
      # Create input
      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f32)

      # Create quantized weight (simulated)
      qt = create_quantized_weight([4, 8])  # [in, out]

      # This should transparently use quantized_matmul
      result = Nx.dot(input, qt)

      assert Nx.shape(result) == {1, 8}
    end

    test "produces same result as dequantize + dot" do
      input = random_input([1, 128])
      weight = random_weight([128, 256])

      # Quantize
      {qw, scales, biases} = EMLX.quantize(weight, 64, 4)
      qt = EMLX.Backend.from_quantized(qw, scales, biases)

      # Dequantize for reference
      dequantized = EMLX.dequantize(qw, scales, biases, 64, 4)

      # Compare results
      result_quantized = Nx.dot(input, qt)
      result_reference = Nx.dot(input, dequantized)

      # Should be close (quantization introduces small errors)
      assert_all_close(result_quantized, result_reference, atol: 0.1)
    end
  end

  describe "LoRA with quantized base" do
    test "LoRA addition works with quantized tensors" do
      input = random_input([1, 128])

      # Quantized base weight
      base_qt = create_quantized_weight([128, 256])

      # LoRA matrices (not quantized)
      lora_a = random_weight([128, 8])
      lora_b = random_weight([8, 256])
      scaling = 20.0

      # Base output (uses quantized_matmul transparently)
      base_out = Nx.dot(input, base_qt)

      # LoRA output
      lora_out = input |> Nx.dot(lora_a) |> Nx.dot(lora_b) |> Nx.multiply(scaling)

      # Combined
      result = Nx.add(base_out, lora_out)

      assert Nx.shape(result) == {1, 256}
    end
  end
end
```

### Integration Tests

```elixir
defmodule EMLX.IntegrationTest do
  use ExUnit.Case

  @moduletag :integration

  describe "real model loading" do
    @tag :slow
    test "loads Qwen3-8B-4bit and runs inference" do
      model_path = System.get_env("QWEN3_MODEL_PATH") || skip()

      # Load with quantized tensors
      {:ok, model} = load_quantized_model(model_path)

      # Simple forward pass
      input_ids = Nx.tensor([[1, 2, 3, 4]])
      {logits, _cache} = forward(input_ids, model)

      assert Nx.shape(logits) == {1, 4, 152064}  # vocab size
    end
  end
end
```

## Migration Path

### For EMLX Users

**Before (current):**
```elixir
# Must use EMLX-specific functions
result = EMLX.quantized_matmul(x, w, s, b, true, 64, 4)
```

**After (proposed):**
```elixir
# Load once as quantized
qt = EMLX.Backend.from_quantized(w, s, b, bits: 4, group_size: 64)

# Use standard Nx everywhere
result = Nx.dot(x, qt)
```

### For Bumblebee

**Before (current bobby_posts):**
- Custom model definitions with EMLX calls
- Can't upstream to Bumblebee

**After (proposed):**
- Standard Bumblebee model definitions unchanged
- Only loader needs quantization awareness
- Can upstream Qwen3/LLaMA quantized support to Bumblebee

## Open Questions

1. **Tensor Protocol**: Should `EMLX.QuantizedTensor` implement `Nx.Tensor` protocol or be a wrapper?

2. **Serialization**: How to serialize quantized tensors? Keep as separate weight/scales/biases in safetensors?

3. **Mixed Precision**: What happens when mixing quantized and non-quantized tensors in operations?

4. **Other Ops**: Which other Nx operations should be quantization-aware? (matmul, linear, etc.)

5. **MLX Native**: Paulo mentioned "something to configure at the MLX level to do this branching automatically" - investigate MLX's type system.

## Timeline

1. **Week 1**: Implement `EMLX.QuantizedTensor` and `from_quantized`
2. **Week 2**: Override `Nx.dot` in EMLX backend
3. **Week 3**: Add tests, edge cases
4. **Week 4**: Update bobby_posts to use new API, validate
5. **Week 5**: PR to EMLX with full test coverage
6. **Week 6+**: Work with Bumblebee team on loader integration

## References

- [EMLX PR #95](https://github.com/elixir-nx/emlx/pull/95)
- [MLX Quantization Docs](https://ml-explore.github.io/mlx/build/html/python/quantization.html)
- [bobby_posts Implementation](https://github.com/notactuallytreyanastasio/bobby_posts)
- [bumblebee_quantized Package](https://github.com/notactuallytreyanastasio/bumblebee_quantized)
