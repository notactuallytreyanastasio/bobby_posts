defmodule BobbyPosts.Qwen3.Layers do
  @moduledoc """
  Core layer implementations for Qwen3 model.

  Includes:
  - RMSNorm: Root Mean Square Layer Normalization
  - RoPE: Rotary Position Embeddings
  - SwiGLU: Gated Linear Unit with SiLU activation
  """

  import Nx.Defn

  @doc """
  Root Mean Square Layer Normalization.

  Unlike LayerNorm, RMSNorm doesn't center the inputs and only scales them.
  Formula: x * weight / sqrt(mean(x^2) + eps)
  """
  deftransform rms_norm(x, weight, opts \\ []) do
    eps = Keyword.get(opts, :eps, 1.0e-6)
    do_rms_norm(x, weight, eps)
  end

  defnp do_rms_norm(x, weight, eps) do
    # x: [..., hidden_size]
    # weight: [hidden_size]
    variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
    x_normalized = x * Nx.rsqrt(variance + eps)
    x_normalized * weight
  end

  @doc """
  Computes RoPE (Rotary Position Embeddings) frequency tensor.

  Returns cos and sin tensors for applying rotary embeddings.
  This is NOT a defn - it uses static shapes determined at runtime.
  """
  def compute_rope_freqs(seq_len, head_dim, opts \\ []) do
    base = Keyword.get(opts, :rope_theta, 1_000_000.0)
    dim_pairs = div(head_dim, 2)

    # Compute inverse frequencies
    # inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    inv_freq_exponents =
      Nx.iota({dim_pairs})
      |> Nx.as_type(:f32)
      |> Nx.multiply(2.0 / head_dim)

    inv_freq = Nx.pow(base, Nx.negate(inv_freq_exponents))

    # Create position indices
    positions = Nx.iota({seq_len}) |> Nx.as_type(:f32)

    # freqs = positions @ inv_freq  -> [seq_len, dim/2]
    freqs = Nx.outer(positions, inv_freq)

    # Return cos and sin
    cos_freqs = Nx.cos(freqs)
    sin_freqs = Nx.sin(freqs)

    {cos_freqs, sin_freqs}
  end

  @doc """
  Applies rotary position embeddings to query and key tensors.

  Q has shape [batch, num_heads, seq_len, head_dim]
  K has shape [batch, num_kv_heads, seq_len, head_dim] (may differ from Q in GQA)
  """
  def apply_rope(q, k, cos_freqs, sin_freqs) do
    # Apply RoPE to Q and K separately (they may have different num_heads for GQA)
    q_rotated = apply_rope_single(q, cos_freqs, sin_freqs)
    k_rotated = apply_rope_single(k, cos_freqs, sin_freqs)
    {q_rotated, k_rotated}
  end

  @doc """
  Applies rotary position embeddings to a single tensor.
  """
  def apply_rope_single(x, cos_freqs, sin_freqs) do
    # x: [batch, num_heads, seq_len, head_dim]
    # cos_freqs, sin_freqs: [seq_len, head_dim/2]
    {batch, heads, seq_len, head_dim} = Nx.shape(x)
    half_dim = div(head_dim, 2)

    # Reshape cos/sin for broadcasting: [1, 1, seq_len, half_dim]
    cos_freqs = cos_freqs |> Nx.slice([0, 0], [seq_len, half_dim])
    sin_freqs = sin_freqs |> Nx.slice([0, 0], [seq_len, half_dim])
    cos_freqs = Nx.reshape(cos_freqs, {1, 1, seq_len, half_dim})
    sin_freqs = Nx.reshape(sin_freqs, {1, 1, seq_len, half_dim})

    # Split into first half and second half of head_dim
    x_first = Nx.slice(x, [0, 0, 0, 0], [batch, heads, seq_len, half_dim])
    x_second = Nx.slice(x, [0, 0, 0, half_dim], [batch, heads, seq_len, half_dim])

    # Apply rotation
    # rotated_first = first * cos - second * sin
    # rotated_second = first * sin + second * cos
    rotated_first = Nx.subtract(Nx.multiply(x_first, cos_freqs), Nx.multiply(x_second, sin_freqs))
    rotated_second = Nx.add(Nx.multiply(x_first, sin_freqs), Nx.multiply(x_second, cos_freqs))

    # Concatenate back
    Nx.concatenate([rotated_first, rotated_second], axis: -1)
  end

  @doc """
  SwiGLU activation: gate * silu(x)
  Used in Qwen3 MLP.
  """
  def swiglu(gate, up) do
    # SiLU activation: x * sigmoid(x)
    gate_activated = Nx.multiply(gate, Nx.sigmoid(gate))
    Nx.multiply(gate_activated, up)
  end

  @doc """
  Softmax attention scores.
  """
  def softmax(x, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    max_val = Nx.reduce_max(x, axes: [axis], keep_axes: true)
    exp_x = Nx.exp(Nx.subtract(x, max_val))
    sum_exp = Nx.sum(exp_x, axes: [axis], keep_axes: true)
    Nx.divide(exp_x, sum_exp)
  end

  @doc """
  Creates a causal attention mask.
  """
  def causal_mask(seq_len) do
    # Create lower triangular mask
    # 1 = attend, 0 = mask out
    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})
    Nx.greater_equal(rows, cols) |> Nx.as_type(:f32)
  end

  @doc """
  Applies causal mask to attention scores.
  """
  def apply_causal_mask(scores, mask) do
    # scores: [batch, heads, q_len, kv_len]
    # mask: [q_len, kv_len] or [1, 1, q_len, kv_len]
    {q_len, kv_len} = {Nx.axis_size(mask, 0), Nx.axis_size(mask, 1)}
    mask = Nx.reshape(mask, {1, 1, q_len, kv_len})
    # Replace masked positions with large negative value
    Nx.add(Nx.multiply(scores, mask), Nx.multiply(Nx.subtract(1.0, mask), -1.0e9))
  end
end
