defmodule BobbyPosts.Qwen3.Model do
  @moduledoc """
  Qwen3 Causal Language Model implementation.

  Architecture:
  - Embedding layer (quantized)
  - 36 transformer layers with GQA and SwiGLU MLP
  - Final RMSNorm
  - LM head (quantized, tied to embeddings)
  """

  require Logger
  alias BobbyPosts.Qwen3.{Attention, Layers}

  @doc """
  Forward pass through the model.

  ## Parameters
    - input_ids: [batch, seq_len] token IDs
    - model: Loaded model weights from QuantizedLoader
    - opts: Options including kv_cache, past_positions, etc.

  ## Returns
    {logits, updated_kv_cache}
  """
  def forward(input_ids, model, opts \\ []) do
    config = model.config
    kv_cache = opts[:kv_cache]
    past_len = opts[:past_len] || 0

    {_batch, seq_len} = Nx.shape(input_ids)

    # Compute RoPE frequencies
    head_dim = div(config["hidden_size"], config["num_attention_heads"])
    total_len = past_len + seq_len
    {cos_freqs, sin_freqs} = Layers.compute_rope_freqs(total_len, head_dim, rope_theta: config["rope_theta"])

    # Slice for current positions only
    cos_freqs = Nx.slice(cos_freqs, [past_len, 0], [seq_len, div(head_dim, 2)])
    sin_freqs = Nx.slice(sin_freqs, [past_len, 0], [seq_len, div(head_dim, 2)])

    # Embedding lookup via dequantized embedding
    hidden_states = embedding_lookup(input_ids, model.embed_tokens, config)

    # Process through transformer layers
    {hidden_states, new_kv_caches} =
      model.layers
      |> Enum.with_index()
      |> Enum.reduce({hidden_states, []}, fn {layer_weights, idx}, {h, caches} ->
        layer_cache = if kv_cache, do: Enum.at(kv_cache, idx), else: nil

        {h_new, cache} =
          transformer_layer(h, layer_weights, config,
            cos_freqs: cos_freqs,
            sin_freqs: sin_freqs,
            kv_cache: layer_cache
          )

        {h_new, caches ++ [cache]}
      end)

    # Final layer norm
    hidden_states = Layers.rms_norm(hidden_states, model.norm, eps: config["rms_norm_eps"])

    # LM head projection to vocabulary
    logits = lm_head(hidden_states, model.lm_head)

    {logits, new_kv_caches}
  end

  @doc """
  Single transformer layer forward pass.
  """
  def transformer_layer(hidden_states, layer_weights, config, opts) do
    cos_freqs = opts[:cos_freqs]
    sin_freqs = opts[:sin_freqs]
    kv_cache = opts[:kv_cache]

    # Pre-attention norm
    normed = Layers.rms_norm(hidden_states, layer_weights.input_layernorm, eps: config["rms_norm_eps"])

    # Self-attention with residual
    {attn_output, new_cache} =
      Attention.forward(normed, layer_weights, config,
        cos_freqs: cos_freqs,
        sin_freqs: sin_freqs,
        kv_cache: kv_cache
      )

    hidden_states = Nx.add(hidden_states, attn_output)

    # Pre-MLP norm
    normed = Layers.rms_norm(hidden_states, layer_weights.post_attention_layernorm, eps: config["rms_norm_eps"])

    # MLP with SwiGLU
    mlp_output = mlp_forward(normed, layer_weights.mlp)
    hidden_states = Nx.add(hidden_states, mlp_output)

    {hidden_states, new_cache}
  end

  @doc """
  MLP forward pass with SwiGLU activation.
  """
  def mlp_forward(x, mlp_weights) do
    gate = Attention.quantized_linear(x, mlp_weights.gate_proj)
    up = Attention.quantized_linear(x, mlp_weights.up_proj)

    # SwiGLU: down(gate * silu(up))
    hidden = Layers.swiglu(gate, up)

    Attention.quantized_linear(hidden, mlp_weights.down_proj)
  end

  @doc """
  Embedding lookup with dequantization.

  For quantized embeddings, we dequantize the relevant rows.
  """
  def embedding_lookup(input_ids, embed_weights, _config) do
    %{weight: w, scales: s, biases: b} = embed_weights

    # Get raw EMLX tensors
    w_raw = EMLX.Backend.from_nx(w)
    s_raw = EMLX.Backend.from_nx(s)
    b_raw = EMLX.Backend.from_nx(b)

    # Dequantize the entire embedding matrix
    # TODO: For efficiency, only dequantize needed rows
    full_embed = EMLX.dequantize(w_raw, s_raw, b_raw, 64, 4)
    embed_nx = EMLX.Backend.to_nx(full_embed)

    # Gather embeddings for input_ids
    Nx.take(embed_nx, input_ids, axis: 0)
  end

  @doc """
  LM head projection (tied to embeddings).
  """
  def lm_head(hidden_states, lm_head_weights) do
    Attention.quantized_linear(hidden_states, lm_head_weights)
  end

  @doc """
  Get next token logits for generation.

  Only returns logits for the last position.
  """
  def get_next_token_logits(input_ids, model, opts \\ []) do
    {logits, kv_cache} = forward(input_ids, model, opts)

    # Take logits for last position only
    {_batch, seq_len, _vocab} = Nx.shape(logits)
    last_logits = Nx.slice(logits, [0, seq_len - 1, 0], [1, 1, Nx.axis_size(logits, 2)])
    last_logits = Nx.squeeze(last_logits, axes: [1])

    {last_logits, kv_cache}
  end
end
