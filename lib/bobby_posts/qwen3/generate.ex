defmodule BobbyPosts.Qwen3.Generate do
  @moduledoc """
  Text generation with KV cache for efficient autoregressive decoding.
  """

  require Logger
  alias BobbyPosts.Qwen3.Model

  @doc """
  Generates text autoregressively using greedy decoding.

  ## Parameters
    - input_ids: Initial token IDs [batch, seq_len]
    - model: Loaded model weights
    - opts: Options including :max_tokens, :eos_token_id, etc.

  ## Returns
    List of generated token IDs
  """
  def generate(input_ids, model, opts \\ []) do
    max_tokens = Keyword.get(opts, :max_tokens, 50)
    eos_token_id = Keyword.get(opts, :eos_token_id, nil)
    verbose = Keyword.get(opts, :verbose, false)
    temperature = Keyword.get(opts, :temperature, 0.7)
    top_p = Keyword.get(opts, :top_p, 0.9)
    adapters = Keyword.get(opts, :adapters, nil)

    # Initial forward pass to get first logits and KV cache
    {_batch, input_len} = Nx.shape(input_ids)

    if verbose do
      Logger.info("Starting generation with #{input_len} input tokens, max_tokens=#{max_tokens}, temp=#{temperature}")
      if adapters, do: Logger.info("Using LoRA adapters with scaling=#{adapters.scaling}")
    end

    {logits, kv_cache} = Model.get_next_token_logits(input_ids, model, kv_cache: nil, past_len: 0, adapters: adapters)

    # Get first token
    first_token = if temperature == 0 do
      sample_greedy(logits)
    else
      sample_with_temperature(logits, temperature, top_p)
    end

    if verbose do
      Logger.info("Generated token 1: #{Nx.to_number(first_token)}")
    end

    # Generate remaining tokens
    generate_loop(
      [first_token],
      kv_cache,
      model,
      input_len,
      max_tokens - 1,
      eos_token_id,
      verbose,
      temperature,
      top_p,
      adapters
    )
  end

  defp generate_loop(tokens, _kv_cache, _model, _past_len, 0, _eos, _verbose, _temp, _top_p, _adapters) do
    # Reached max tokens
    Enum.reverse(tokens) |> Enum.map(&Nx.to_number/1)
  end

  defp generate_loop(tokens, kv_cache, model, past_len, remaining, eos_token_id, verbose, temperature, top_p, adapters) do
    [last_token | _] = tokens
    last_token_id = Nx.to_number(last_token)

    # Check for EOS
    if eos_token_id && last_token_id == eos_token_id do
      Enum.reverse(tokens) |> Enum.map(&Nx.to_number/1)
    else
      # Forward pass with just the last token
      input = Nx.reshape(last_token, {1, 1})
      new_past_len = past_len + length(tokens)

      {logits, new_kv_cache} =
        Model.get_next_token_logits(input, model, kv_cache: kv_cache, past_len: new_past_len - 1, adapters: adapters)

      next_token = if temperature == 0 do
        sample_greedy(logits)
      else
        sample_with_temperature(logits, temperature, top_p)
      end

      if verbose && rem(length(tokens), 10) == 0 do
        Logger.info("Generated #{length(tokens)} tokens...")
      end

      generate_loop(
        [next_token | tokens],
        new_kv_cache,
        model,
        past_len,
        remaining - 1,
        eos_token_id,
        verbose,
        temperature,
        top_p,
        adapters
      )
    end
  end

  @doc """
  Greedy sampling - select the token with highest probability.
  """
  def sample_greedy(logits) do
    # logits: [batch, vocab_size] or [vocab_size]
    # Returns a scalar tensor
    logits
    |> Nx.argmax(axis: -1)
    |> Nx.squeeze()
  end

  @doc """
  Temperature-based sampling with top-p filtering.
  """
  def sample_with_temperature(logits, temperature, top_p) do
    # Squeeze to 1D if needed
    logits = Nx.squeeze(logits)

    # Apply temperature
    scaled_logits = Nx.divide(logits, temperature)

    # Softmax to get probabilities
    max_logit = Nx.reduce_max(scaled_logits)
    exp_logits = Nx.exp(Nx.subtract(scaled_logits, max_logit))
    probs = Nx.divide(exp_logits, Nx.sum(exp_logits))

    # Convert to list for CPU-based top-p sampling
    probs_list = Nx.to_flat_list(probs)

    # Create list of {prob, index} and sort descending by prob
    indexed_probs = probs_list
      |> Enum.with_index()
      |> Enum.sort_by(fn {prob, _idx} -> -prob end)

    # Apply top-p filtering and sample
    {filtered_probs, filtered_indices} =
      indexed_probs
      |> Enum.reduce_while({[], [], 0.0}, fn {prob, idx}, {probs_acc, idx_acc, cumsum} ->
        new_cumsum = cumsum + prob
        if cumsum >= top_p and length(probs_acc) > 0 do
          {:halt, {probs_acc, idx_acc, new_cumsum}}
        else
          {:cont, {[prob | probs_acc], [idx | idx_acc], new_cumsum}}
        end
      end)
      |> then(fn {probs_acc, idx_acc, _cumsum} ->
        {Enum.reverse(probs_acc), Enum.reverse(idx_acc)}
      end)

    # Renormalize
    total = Enum.sum(filtered_probs)
    normalized_probs = Enum.map(filtered_probs, &(&1 / total))

    # Sample using inverse CDF
    random_val = :rand.uniform()
    selected_idx = sample_from_probs(normalized_probs, filtered_indices, random_val)

    Nx.tensor(selected_idx)
  end

  defp sample_from_probs(probs, indices, random_val) do
    {_cumsum, selected} =
      Enum.zip(probs, indices)
      |> Enum.reduce_while({0.0, hd(indices)}, fn {prob, idx}, {cumsum, _} ->
        new_cumsum = cumsum + prob
        if new_cumsum >= random_val do
          {:halt, {new_cumsum, idx}}
        else
          {:cont, {new_cumsum, idx}}
        end
      end)
    selected
  end

  @doc """
  Top-p (nucleus) sampling.
  """
  def sample_top_p(logits, p \\ 0.9, temperature \\ 1.0) do
    # Apply temperature
    scaled_logits = Nx.divide(logits, temperature)

    # Softmax to get probabilities
    probs = Nx.exp(Nx.subtract(scaled_logits, Nx.reduce_max(scaled_logits, axes: [-1], keep_axes: true)))
    probs = Nx.divide(probs, Nx.sum(probs, axes: [-1], keep_axes: true))

    # Sort probabilities in descending order
    sorted_indices = Nx.argsort(probs, axis: -1, direction: :desc)
    sorted_probs = Nx.take_along_axis(probs, sorted_indices, axis: -1)

    # Compute cumulative sum
    cumsum = Nx.cumulative_sum(sorted_probs, axis: -1)

    # Create mask for tokens within top-p
    mask = Nx.less(cumsum, p) |> Nx.as_type(:f32)
    # Always include at least the first token
    mask = Nx.put_slice(mask, [0, 0], Nx.tensor([[1.0]]))

    # Apply mask
    masked_probs = Nx.multiply(sorted_probs, mask)
    masked_probs = Nx.divide(masked_probs, Nx.sum(masked_probs, axes: [-1], keep_axes: true))

    # Sample from the distribution
    # For now, just take the argmax of masked distribution (simplified)
    sorted_idx = Nx.argmax(masked_probs, axis: -1)

    # Map back to original indices
    Nx.take_along_axis(sorted_indices, Nx.reshape(sorted_idx, {1, 1}), axis: -1)
    |> Nx.squeeze()
  end

  @doc """
  Generates text given a tokenized prompt.
  Returns the full sequence (input + generated).
  """
  def generate_text(prompt_tokens, model, tokenizer, opts \\ []) do
    max_tokens = Keyword.get(opts, :max_tokens, 50)
    verbose = Keyword.get(opts, :verbose, false)

    # Get special token IDs from tokenizer if available
    eos_token_id = Keyword.get(opts, :eos_token_id, 151643)  # Qwen3 default

    if verbose do
      Logger.info("Generating with #{length(prompt_tokens)} prompt tokens...")
    end

    # Convert to tensor
    input_ids = Nx.tensor([prompt_tokens], type: :s32)

    # Generate
    generated = generate(input_ids, model,
      max_tokens: max_tokens,
      eos_token_id: eos_token_id,
      verbose: verbose
    )

    # Decode tokens
    full_tokens = prompt_tokens ++ generated

    case tokenizer do
      nil ->
        # Return just the token IDs
        {:ok, %{tokens: full_tokens, generated: generated}}

      tokenizer_module when is_atom(tokenizer_module) ->
        # Decode using provided tokenizer module
        text = tokenizer_module.decode(full_tokens)
        {:ok, %{text: text, tokens: full_tokens, generated: generated}}

      _ ->
        {:ok, %{tokens: full_tokens, generated: generated}}
    end
  end
end
