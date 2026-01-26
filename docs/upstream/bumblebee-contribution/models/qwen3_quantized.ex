defmodule Bumblebee.Text.Qwen3Quantized do
  @moduledoc """
  Qwen3 Quantized Causal Language Model for Apple Silicon.

  This is a separate quantized model definition that uses EMLX's
  `quantized_matmul` for efficient 4-bit inference on Apple Silicon.

  ## Architecture

  - Embedding layer (quantized, dequantized for lookup)
  - 36 transformer layers with GQA and SwiGLU MLP
  - Final RMSNorm
  - LM head (quantized, may be tied to embeddings)

  ## Features

  - 4-bit quantized weights (MLX format)
  - Grouped Query Attention (GQA)
  - Rotary Position Embeddings (RoPE)
  - KV cache for efficient autoregressive generation
  - Runtime LoRA adapter support

  ## Example

      {:ok, model} = Bumblebee.QuantizedLoader.load_model(
        "/path/to/Qwen3-8B-4bit",
        architecture: :qwen3
      )

      {:ok, adapter} = Bumblebee.Adapters.load("/path/to/adapter")

      {logits, kv_cache} = Bumblebee.Text.Qwen3Quantized.forward(
        input_ids,
        model,
        adapter: adapter
      )

  ## Requirements

  - Apple Silicon Mac (M1/M2/M3/M4)
  - EMLX backend with quantization ops
  """

  import Nx.Defn
  require Logger

  # ============================================================================
  # Forward Pass
  # ============================================================================

  @doc """
  Forward pass through the Qwen3 model.

  ## Parameters

    * `input_ids` - Token IDs with shape `[batch, seq_len]`
    * `model` - Loaded model weights from `Bumblebee.QuantizedLoader`
    * `opts` - Options:
      * `:kv_cache` - KV cache from previous forward pass
      * `:past_len` - Length of cached sequence
      * `:adapter` - LoRA adapter from `Bumblebee.Adapters`

  ## Returns

  `{logits, kv_cache}` where:
    * `logits` - Output logits with shape `[batch, seq_len, vocab_size]`
    * `kv_cache` - Updated KV cache for next forward pass
  """
  def forward(input_ids, model, opts \\ []) do
    config = model.config
    kv_cache = opts[:kv_cache]
    past_len = opts[:past_len] || 0
    adapter = opts[:adapter]
    lora_scaling = if adapter, do: adapter.scaling, else: 0.0

    {_batch, seq_len} = Nx.shape(input_ids)

    # Compute RoPE frequencies
    head_dim = div(config["hidden_size"], config["num_attention_heads"])
    total_len = past_len + seq_len
    {cos_freqs, sin_freqs} = compute_rope_freqs(total_len, head_dim, config["rope_theta"])

    # Slice for current positions only
    cos_freqs = Nx.slice(cos_freqs, [past_len, 0], [seq_len, div(head_dim, 2)])
    sin_freqs = Nx.slice(sin_freqs, [past_len, 0], [seq_len, div(head_dim, 2)])

    # Embedding lookup via dequantized embedding
    hidden_states = embedding_lookup(input_ids, model.embed_tokens)

    # Process through transformer layers
    {hidden_states, new_kv_caches} =
      model.layers
      |> Enum.with_index()
      |> Enum.reduce({hidden_states, []}, fn {layer_weights, idx}, {h, caches} ->
        layer_cache = if kv_cache, do: Enum.at(kv_cache, idx), else: nil
        layer_adapter = if adapter, do: Bumblebee.Adapters.get_layer_adapter(adapter, idx), else: nil

        {h_new, cache} = transformer_layer(h, layer_weights, config,
          cos_freqs: cos_freqs,
          sin_freqs: sin_freqs,
          kv_cache: layer_cache,
          adapter: layer_adapter,
          lora_scaling: lora_scaling
        )

        {h_new, caches ++ [cache]}
      end)

    # Final layer norm
    hidden_states = rms_norm(hidden_states, model.norm, config["rms_norm_eps"])

    # LM head projection to vocabulary
    logits = quantized_linear(hidden_states, model.lm_head)

    {logits, new_kv_caches}
  end

  @doc """
  Get next token logits for generation (last position only).
  """
  def get_next_token_logits(input_ids, model, opts \\ []) do
    {logits, kv_cache} = forward(input_ids, model, opts)

    # Take logits for last position only
    {_batch, seq_len, _vocab} = Nx.shape(logits)
    last_logits = Nx.slice(logits, [0, seq_len - 1, 0], [1, 1, Nx.axis_size(logits, 2)])
    last_logits = Nx.squeeze(last_logits, axes: [1])

    {last_logits, kv_cache}
  end

  # ============================================================================
  # Transformer Layer
  # ============================================================================

  defp transformer_layer(hidden_states, layer_weights, config, opts) do
    cos_freqs = opts[:cos_freqs]
    sin_freqs = opts[:sin_freqs]
    kv_cache = opts[:kv_cache]
    adapter = opts[:adapter]
    lora_scaling = opts[:lora_scaling] || 0.0
    eps = config["rms_norm_eps"]

    # Pre-attention norm
    normed = rms_norm(hidden_states, layer_weights.input_layernorm, eps)

    # Self-attention with residual
    {attn_output, new_cache} = attention_forward(normed, layer_weights, config,
      cos_freqs: cos_freqs,
      sin_freqs: sin_freqs,
      kv_cache: kv_cache,
      adapter: adapter,
      lora_scaling: lora_scaling
    )
    hidden_states = Nx.add(hidden_states, attn_output)

    # Pre-MLP norm
    normed = rms_norm(hidden_states, layer_weights.post_attention_layernorm, eps)

    # MLP with SwiGLU
    mlp_adapter = if adapter, do: adapter.mlp, else: nil
    mlp_output = mlp_forward(normed, layer_weights.mlp, mlp_adapter, lora_scaling)
    hidden_states = Nx.add(hidden_states, mlp_output)

    {hidden_states, new_cache}
  end

  # ============================================================================
  # Attention
  # ============================================================================

  defp attention_forward(hidden_states, layer_weights, config, opts) do
    cos_freqs = opts[:cos_freqs]
    sin_freqs = opts[:sin_freqs]
    kv_cache = opts[:kv_cache]
    adapter = opts[:adapter]
    scaling = opts[:lora_scaling] || 0.0

    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = div(hidden_size, num_heads)
    num_kv_groups = div(num_heads, num_kv_heads)
    eps = config["rms_norm_eps"]

    {batch, seq_len, _} = Nx.shape(hidden_states)

    attn_adapter = if adapter, do: adapter.self_attn, else: nil

    # Project to Q, K, V
    q = quantized_linear_with_lora(hidden_states, layer_weights.self_attn.q_proj,
      attn_adapter && attn_adapter.q_proj, scaling)
    k = quantized_linear_with_lora(hidden_states, layer_weights.self_attn.k_proj,
      attn_adapter && attn_adapter.k_proj, scaling)
    v = quantized_linear_with_lora(hidden_states, layer_weights.self_attn.v_proj,
      attn_adapter && attn_adapter.v_proj, scaling)

    # Reshape to [batch, seq, num_heads, head_dim]
    q = Nx.reshape(q, {batch, seq_len, num_heads, head_dim})
    k = Nx.reshape(k, {batch, seq_len, num_kv_heads, head_dim})

    # Apply Q/K RMSNorm (Qwen3 specific)
    q = apply_qk_norm(q, layer_weights.self_attn.q_norm, eps)
    k = apply_qk_norm(k, layer_weights.self_attn.k_norm, eps)

    # Transpose to [batch, num_heads, seq, head_dim]
    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Apply RoPE
    {q, k} = apply_rope(q, k, cos_freqs, sin_freqs)

    # Handle KV cache
    {k, v, updated_cache} = handle_kv_cache(k, v, kv_cache)

    # Repeat K, V for GQA
    k = repeat_kv(k, num_kv_groups)
    v = repeat_kv(v, num_kv_groups)

    # Compute attention
    scale = :math.sqrt(head_dim)
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)

    # Apply causal mask
    kv_len = Nx.axis_size(k, 2)
    mask = causal_mask(kv_len)
    mask = if kv_cache, do: Nx.slice(mask, [kv_len - seq_len, 0], [seq_len, kv_len]), else: mask
    scores = apply_causal_mask(scores, mask)

    # Softmax and attend
    attn_weights = softmax(scores)
    attn_output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back
    attn_output = attn_output
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq_len, hidden_size})

    # Output projection
    output = quantized_linear_with_lora(attn_output, layer_weights.self_attn.o_proj,
      attn_adapter && attn_adapter.o_proj, scaling)

    {output, updated_cache}
  end

  defp handle_kv_cache(k, v, nil), do: {k, v, {k, v}}
  defp handle_kv_cache(k, v, {cached_k, cached_v}) do
    k = Nx.concatenate([cached_k, k], axis: 2)
    v = Nx.concatenate([cached_v, v], axis: 2)
    {k, v, {k, v}}
  end

  # ============================================================================
  # MLP
  # ============================================================================

  defp mlp_forward(x, mlp_weights, mlp_adapter, lora_scaling) do
    gate = quantized_linear_with_lora(x, mlp_weights.gate_proj,
      mlp_adapter && mlp_adapter.gate_proj, lora_scaling)
    up = quantized_linear_with_lora(x, mlp_weights.up_proj,
      mlp_adapter && mlp_adapter.up_proj, lora_scaling)

    # SwiGLU: silu(gate) * up
    hidden = swiglu(gate, up)

    quantized_linear_with_lora(hidden, mlp_weights.down_proj,
      mlp_adapter && mlp_adapter.down_proj, lora_scaling)
  end

  # ============================================================================
  # Core Operations
  # ============================================================================

  @doc """
  Quantized linear projection using EMLX.quantized_matmul.
  """
  def quantized_linear(x, %{weight: w, scales: s, biases: b}) do
    x_raw = EMLX.Backend.from_nx(x)
    w_raw = EMLX.Backend.from_nx(w)
    s_raw = EMLX.Backend.from_nx(s)
    b_raw = EMLX.Backend.from_nx(b)

    result = EMLX.quantized_matmul(x_raw, w_raw, s_raw, b_raw, true, 64, 4)
    EMLX.Backend.to_nx(result)
  end

  @doc """
  Quantized linear with optional LoRA adapter.
  """
  def quantized_linear_with_lora(x, base_weights, nil, _scaling) do
    quantized_linear(x, base_weights)
  end

  def quantized_linear_with_lora(x, base_weights, %{lora_a: lora_a, lora_b: lora_b}, scaling) do
    base_output = quantized_linear(x, base_weights)

    # LoRA: output = base + scaling * (x @ A @ B)
    lora_output = x
      |> Nx.dot([-1], lora_a, [0])
      |> Nx.dot([-1], lora_b, [0])
      |> Nx.multiply(scaling)

    Nx.add(base_output, lora_output)
  end

  @doc """
  Embedding lookup with dequantization.
  """
  def embedding_lookup(input_ids, %{weight: w, scales: s, biases: b}) do
    w_raw = EMLX.Backend.from_nx(w)
    s_raw = EMLX.Backend.from_nx(s)
    b_raw = EMLX.Backend.from_nx(b)

    # Dequantize entire embedding matrix
    full_embed = EMLX.dequantize(w_raw, s_raw, b_raw, 64, 4)
    embed_nx = EMLX.Backend.to_nx(full_embed)

    Nx.take(embed_nx, input_ids, axis: 0)
  end

  # ============================================================================
  # Layer Operations
  # ============================================================================

  deftransform rms_norm(x, weight, eps) do
    do_rms_norm(x, weight, eps)
  end

  defnp do_rms_norm(x, weight, eps) do
    variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
    x_normalized = x * Nx.rsqrt(variance + eps)
    x_normalized * weight
  end

  defp apply_qk_norm(x, weight, eps) do
    {_batch, _seq_len, _heads, head_dim} = Nx.shape(x)
    variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
    x_normalized = Nx.multiply(x, Nx.rsqrt(Nx.add(variance, eps)))
    weight = Nx.reshape(weight, {1, 1, 1, head_dim})
    Nx.multiply(x_normalized, weight)
  end

  defp swiglu(gate, up) do
    gate_activated = Nx.multiply(gate, Nx.sigmoid(gate))
    Nx.multiply(gate_activated, up)
  end

  defp softmax(x) do
    max_val = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp_x = Nx.exp(Nx.subtract(x, max_val))
    sum_exp = Nx.sum(exp_x, axes: [-1], keep_axes: true)
    Nx.divide(exp_x, sum_exp)
  end

  # ============================================================================
  # RoPE
  # ============================================================================

  defp compute_rope_freqs(seq_len, head_dim, base) do
    dim_pairs = div(head_dim, 2)

    inv_freq_exponents = Nx.iota({dim_pairs})
      |> Nx.as_type(:f32)
      |> Nx.multiply(2.0 / head_dim)

    inv_freq = Nx.pow(base, Nx.negate(inv_freq_exponents))
    positions = Nx.iota({seq_len}) |> Nx.as_type(:f32)
    freqs = Nx.outer(positions, inv_freq)

    {Nx.cos(freqs), Nx.sin(freqs)}
  end

  defp apply_rope(q, k, cos_freqs, sin_freqs) do
    {apply_rope_single(q, cos_freqs, sin_freqs), apply_rope_single(k, cos_freqs, sin_freqs)}
  end

  defp apply_rope_single(x, cos_freqs, sin_freqs) do
    {batch, heads, seq_len, head_dim} = Nx.shape(x)
    half_dim = div(head_dim, 2)

    cos_freqs = cos_freqs |> Nx.slice([0, 0], [seq_len, half_dim]) |> Nx.reshape({1, 1, seq_len, half_dim})
    sin_freqs = sin_freqs |> Nx.slice([0, 0], [seq_len, half_dim]) |> Nx.reshape({1, 1, seq_len, half_dim})

    x_first = Nx.slice(x, [0, 0, 0, 0], [batch, heads, seq_len, half_dim])
    x_second = Nx.slice(x, [0, 0, 0, half_dim], [batch, heads, seq_len, half_dim])

    rotated_first = Nx.subtract(Nx.multiply(x_first, cos_freqs), Nx.multiply(x_second, sin_freqs))
    rotated_second = Nx.add(Nx.multiply(x_first, sin_freqs), Nx.multiply(x_second, cos_freqs))

    Nx.concatenate([rotated_first, rotated_second], axis: -1)
  end

  # ============================================================================
  # Masking
  # ============================================================================

  defp causal_mask(seq_len) do
    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})
    Nx.greater_equal(rows, cols) |> Nx.as_type(:f32)
  end

  defp apply_causal_mask(scores, mask) do
    {q_len, kv_len} = {Nx.axis_size(mask, 0), Nx.axis_size(mask, 1)}
    mask = Nx.reshape(mask, {1, 1, q_len, kv_len})
    Nx.add(Nx.multiply(scores, mask), Nx.multiply(Nx.subtract(1.0, mask), -1.0e9))
  end

  defp repeat_kv(x, 1), do: x
  defp repeat_kv(x, n_rep) do
    {batch, num_kv_heads, seq_len, head_dim} = Nx.shape(x)

    x
    |> Nx.reshape({batch, num_kv_heads, 1, seq_len, head_dim})
    |> Nx.broadcast({batch, num_kv_heads, n_rep, seq_len, head_dim})
    |> Nx.reshape({batch, num_kv_heads * n_rep, seq_len, head_dim})
  end
end
