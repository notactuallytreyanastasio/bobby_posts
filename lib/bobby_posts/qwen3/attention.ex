defmodule BobbyPosts.Qwen3.Attention do
  @moduledoc """
  Grouped Query Attention (GQA) implementation for Qwen3.

  GQA uses fewer key/value heads than query heads, reducing memory
  and compute while maintaining quality. For Qwen3-8B:
  - Query heads: 32
  - Key/Value heads: 8
  - Head dimension: 128 (4096 / 32)
  """

  alias BobbyPosts.Qwen3.Layers

  @doc """
  Performs grouped query attention with optional KV cache.

  ## Parameters
    - hidden_states: [batch, seq_len, hidden_size]
    - layer_weights: Layer weights including q/k/v/o projections
    - config: Model configuration
    - opts: Options including cos/sin for RoPE, kv_cache, etc.

  ## Returns
    {output, updated_kv_cache}
  """
  def forward(hidden_states, layer_weights, config, opts \\ []) do
    cos_freqs = opts[:cos_freqs]
    sin_freqs = opts[:sin_freqs]
    kv_cache = opts[:kv_cache]
    attention_mask = opts[:attention_mask]
    adapters = opts[:adapters]
    scaling = opts[:lora_scaling] || 0.0

    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = div(hidden_size, num_heads)
    num_kv_groups = div(num_heads, num_kv_heads)

    {batch, seq_len, _} = Nx.shape(hidden_states)
    eps = config["rms_norm_eps"]

    # Get adapter weights for attention if present
    attn_adapters = if adapters, do: adapters.self_attn, else: nil

    # Project to Q, K, V using quantized matmul with optional LoRA
    q = quantized_linear_with_lora(
      hidden_states,
      layer_weights.self_attn.q_proj,
      attn_adapters && attn_adapters.q_proj,
      scaling
    )
    k = quantized_linear_with_lora(
      hidden_states,
      layer_weights.self_attn.k_proj,
      attn_adapters && attn_adapters.k_proj,
      scaling
    )
    v = quantized_linear_with_lora(
      hidden_states,
      layer_weights.self_attn.v_proj,
      attn_adapters && attn_adapters.v_proj,
      scaling
    )

    # Reshape to [batch, seq, num_heads, head_dim] for Q/K normalization
    q = Nx.reshape(q, {batch, seq_len, num_heads, head_dim})
    k = Nx.reshape(k, {batch, seq_len, num_kv_heads, head_dim})

    # Apply Q/K RMSNorm (Qwen3 specific - normalizes per head)
    q = apply_qk_norm(q, layer_weights.self_attn.q_norm, eps)
    k = apply_qk_norm(k, layer_weights.self_attn.k_norm, eps)

    # Transpose to [batch, num_heads, seq, head_dim]
    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Apply RoPE to Q and K
    {q, k} = Layers.apply_rope(q, k, cos_freqs, sin_freqs)

    # Handle KV cache
    {k, v, updated_cache} =
      case kv_cache do
        nil ->
          {k, v, {k, v}}

        {cached_k, cached_v} ->
          # Concatenate with cached values
          k = Nx.concatenate([cached_k, k], axis: 2)
          v = Nx.concatenate([cached_v, v], axis: 2)
          {k, v, {k, v}}
      end

    # Repeat K, V for GQA (expand KV heads to match Q heads)
    k = repeat_kv(k, num_kv_groups)
    v = repeat_kv(v, num_kv_groups)

    # Compute attention
    # Q: [batch, num_heads, q_len, head_dim]
    # K: [batch, num_heads, kv_len, head_dim]
    # scores = Q @ K^T / sqrt(head_dim)
    scale = :math.sqrt(head_dim)
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)

    # Apply causal mask
    scores =
      if attention_mask do
        Layers.apply_causal_mask(scores, attention_mask)
      else
        kv_len = Nx.axis_size(k, 2)
        mask = Layers.causal_mask(kv_len)
        # For cached inference, only mask the query portion
        if kv_cache do
          # Take the last seq_len rows of the mask
          mask = Nx.slice(mask, [kv_len - seq_len, 0], [seq_len, kv_len])
          Layers.apply_causal_mask(scores, mask)
        else
          Layers.apply_causal_mask(scores, mask)
        end
      end

    # Softmax and attend
    attn_weights = Layers.softmax(scores, axis: -1)
    attn_output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden_size]
    attn_output =
      attn_output
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq_len, hidden_size})

    # Output projection with optional LoRA
    output = quantized_linear_with_lora(
      attn_output,
      layer_weights.self_attn.o_proj,
      attn_adapters && attn_adapters.o_proj,
      scaling
    )

    {output, updated_cache}
  end

  @doc """
  Performs quantized linear projection.

  Uses EMLX.dot with QuantizedTensor for transparent 4-bit dispatch.
  """
  def quantized_linear(x, %EMLX.QuantizedTensor{} = qt) do
    # EMLX.dot automatically dispatches to quantized_matmul
    result = EMLX.dot(x, qt)

    # Convert back to Nx tensor
    EMLX.to_nx(result)
  end

  # Legacy support for old {weight, scales, biases} map format
  def quantized_linear(x, %{weight: w, scales: s, biases: b}) do
    # Get raw EMLX tensors
    x_raw = EMLX.Backend.from_nx(x)
    w_raw = EMLX.Backend.from_nx(w)
    s_raw = EMLX.Backend.from_nx(s)
    b_raw = EMLX.Backend.from_nx(b)

    # Run quantized matmul
    result = EMLX.quantized_matmul(x_raw, w_raw, s_raw, b_raw, true, 64, 4)

    # Convert back to Nx tensor
    EMLX.Backend.to_nx(result)
  end

  @doc """
  Performs quantized linear projection with LoRA adapter.

  LoRA formula: output = base_output + (x @ lora_a @ lora_b) * scaling

  ## Parameters
    - x: Input tensor [batch, seq, hidden]
    - base_weights: EMLX.QuantizedTensor with packed 4-bit weights
    - lora_weights: LoRA adapter weights {lora_a, lora_b} or nil
    - scaling: LoRA scaling factor (scale / rank)
  """
  def quantized_linear_with_lora(x, base_weights, nil, _scaling) do
    # No LoRA, just base linear
    quantized_linear(x, base_weights)
  end

  def quantized_linear_with_lora(x, base_weights, %{lora_a: lora_a, lora_b: lora_b}, scaling) do
    # Base output using quantized weights
    base_output = quantized_linear(x, base_weights)

    # LoRA: (x @ lora_a @ lora_b) * scaling
    # lora_a: [input_dim, rank]
    # lora_b: [rank, output_dim]
    # x: [batch, seq, input_dim]
    # Use Nx.LinAlg.dot for batched matrix multiply on GPU
    # Contract last axis of x with first axis of lora_a
    temp = Nx.dot(x, [-1], lora_a, [0])  # [batch, seq, rank]
    lora_output = Nx.dot(temp, [-1], lora_b, [0])  # [batch, seq, output_dim]
    lora_output = Nx.multiply(lora_output, scaling)

    Nx.add(base_output, lora_output)
  end

  # Apply RMSNorm to Q/K per head dimension
  # x: [batch, seq, heads, head_dim]
  # weight: [head_dim]
  defp apply_qk_norm(x, weight, eps) do
    {_batch, _seq_len, _heads, head_dim} = Nx.shape(x)
    # Apply RMSNorm along the head_dim axis
    variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
    x_normalized = Nx.multiply(x, Nx.rsqrt(Nx.add(variance, eps)))
    # Broadcast weight [head_dim] to [1, 1, 1, head_dim]
    weight = Nx.reshape(weight, {1, 1, 1, head_dim})
    Nx.multiply(x_normalized, weight)
  end

  # Repeat KV heads to match number of query heads
  defp repeat_kv(x, n_rep) when n_rep == 1, do: x

  defp repeat_kv(x, n_rep) do
    {batch, num_kv_heads, seq_len, head_dim} = Nx.shape(x)

    x
    |> Nx.reshape({batch, num_kv_heads, 1, seq_len, head_dim})
    |> Nx.broadcast({batch, num_kv_heads, n_rep, seq_len, head_dim})
    |> Nx.reshape({batch, num_kv_heads * n_rep, seq_len, head_dim})
  end
end
