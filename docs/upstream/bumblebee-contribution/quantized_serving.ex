defmodule Bumblebee.Text.QuantizedGeneration do
  @moduledoc """
  Text generation serving for quantized models on Apple Silicon.

  This module provides an `Nx.Serving` compatible interface for
  running text generation with quantized LLMs using EMLX.

  ## Example

      # Load model and tokenizer
      {:ok, model} = Bumblebee.QuantizedLoader.load_model("/path/to/Qwen3-8B-4bit")
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-8B"})

      # Optional: load LoRA adapter
      {:ok, adapter} = Bumblebee.Adapters.load("/path/to/adapter")

      # Create serving
      serving = Bumblebee.Text.QuantizedGeneration.new(model, tokenizer,
        adapter: adapter,
        max_new_tokens: 100,
        temperature: 0.8
      )

      # Run inference
      Nx.Serving.run(serving, "Write a post about Elixir")

  ## Options

    * `:adapter` - LoRA adapter from `Bumblebee.Adapters`
    * `:max_new_tokens` - Maximum tokens to generate (default: 50)
    * `:temperature` - Sampling temperature (default: 0.7)
    * `:top_p` - Nucleus sampling threshold (default: 0.9)
    * `:eos_token_id` - End of sequence token ID (default: auto-detect)

  ## Implementation Notes

  Unlike standard Bumblebee servings, this uses a custom inference
  path optimized for quantized models:

  1. Tokenize input text using Bumblebee tokenizer
  2. Run quantized forward pass with KV caching
  3. Sample next token using temperature/top-p
  4. Repeat until max_tokens or EOS
  5. Decode output tokens

  The quantized model uses EMLX's `quantized_matmul` operation which
  fuses dequantization with matrix multiplication for efficiency.
  """

  require Logger

  @type serving_options :: [
          adapter: Bumblebee.Adapters.adapter() | nil,
          max_new_tokens: pos_integer(),
          temperature: float(),
          top_p: float(),
          eos_token_id: integer() | nil
        ]

  @doc """
  Creates a new quantized generation serving.

  ## Parameters

    * `model` - Loaded quantized model from `Bumblebee.QuantizedLoader`
    * `tokenizer` - Bumblebee tokenizer
    * `opts` - Generation options (see module docs)

  ## Returns

  An `Nx.Serving` struct that can be used with `Nx.Serving.run/2`.
  """
  @spec new(map(), Bumblebee.Tokenizer.t(), serving_options()) :: Nx.Serving.t()
  def new(model, tokenizer, opts \\ []) do
    adapter = Keyword.get(opts, :adapter)
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 50)
    temperature = Keyword.get(opts, :temperature, 0.7)
    top_p = Keyword.get(opts, :top_p, 0.9)
    eos_token_id = Keyword.get(opts, :eos_token_id, detect_eos_token(tokenizer))

    serving_opts = %{
      model: model,
      tokenizer: tokenizer,
      adapter: adapter,
      max_new_tokens: max_new_tokens,
      temperature: temperature,
      top_p: top_p,
      eos_token_id: eos_token_id
    }

    Nx.Serving.new(fn _opts -> serving_opts end, &run_inference/2)
  end

  @doc """
  Runs text generation on a batch of prompts.

  This is called by `Nx.Serving.run/2`.
  """
  @spec run_inference(map(), String.t() | [String.t()]) :: [map()]
  def run_inference(serving_opts, inputs) when is_binary(inputs) do
    run_inference(serving_opts, [inputs])
  end

  def run_inference(serving_opts, inputs) when is_list(inputs) do
    %{
      model: model,
      tokenizer: tokenizer,
      adapter: adapter,
      max_new_tokens: max_new_tokens,
      temperature: temperature,
      top_p: top_p,
      eos_token_id: eos_token_id
    } = serving_opts

    Enum.map(inputs, fn input ->
      generate_single(input, model, tokenizer, adapter, max_new_tokens, temperature, top_p, eos_token_id)
    end)
  end

  defp generate_single(prompt, model, tokenizer, adapter, max_new_tokens, temperature, top_p, eos_token_id) do
    # Tokenize input
    %{input_ids: input_ids} = Bumblebee.apply_tokenizer(tokenizer, prompt)
    input_tokens = Nx.to_flat_list(input_ids)

    # Convert to tensor
    input_tensor = Nx.tensor([input_tokens], type: :s32)
    {_batch, input_len} = Nx.shape(input_tensor)

    # Initial forward pass
    model_opts = if adapter, do: [adapter: adapter], else: []
    {logits, kv_cache} = Bumblebee.Text.Qwen3Quantized.get_next_token_logits(
      input_tensor,
      model,
      Keyword.merge(model_opts, [kv_cache: nil, past_len: 0])
    )

    # Get first token
    first_token = sample_token(logits, temperature, top_p)

    # Generate remaining tokens
    generated_tokens = generate_loop(
      [first_token],
      kv_cache,
      model,
      input_len,
      max_new_tokens - 1,
      eos_token_id,
      temperature,
      top_p,
      adapter
    )

    # Decode generated tokens
    generated_text = Bumblebee.Tokenizer.decode(tokenizer, Nx.tensor(generated_tokens))

    %{
      results: [
        %{
          text: generated_text,
          token_count: length(generated_tokens)
        }
      ]
    }
  end

  defp generate_loop(tokens, _kv_cache, _model, _past_len, 0, _eos, _temp, _top_p, _adapter) do
    Enum.reverse(tokens) |> Enum.map(&Nx.to_number/1)
  end

  defp generate_loop(tokens, kv_cache, model, past_len, remaining, eos_token_id, temperature, top_p, adapter) do
    [last_token | _] = tokens
    last_token_id = Nx.to_number(last_token)

    if eos_token_id && last_token_id == eos_token_id do
      Enum.reverse(tokens) |> Enum.map(&Nx.to_number/1)
    else
      input = Nx.reshape(last_token, {1, 1})
      new_past_len = past_len + length(tokens)

      model_opts = if adapter, do: [adapter: adapter], else: []
      {logits, new_kv_cache} = Bumblebee.Text.Qwen3Quantized.get_next_token_logits(
        input,
        model,
        Keyword.merge(model_opts, [kv_cache: kv_cache, past_len: new_past_len - 1])
      )

      next_token = sample_token(logits, temperature, top_p)

      generate_loop(
        [next_token | tokens],
        new_kv_cache,
        model,
        past_len,
        remaining - 1,
        eos_token_id,
        temperature,
        top_p,
        adapter
      )
    end
  end

  # ============================================================================
  # Sampling
  # ============================================================================

  defp sample_token(logits, temperature, top_p) when temperature == 0 do
    logits |> Nx.argmax(axis: -1) |> Nx.squeeze()
  end

  defp sample_token(logits, temperature, top_p) do
    logits = Nx.squeeze(logits)

    # Apply temperature
    scaled_logits = Nx.divide(logits, temperature)

    # Softmax
    max_logit = Nx.reduce_max(scaled_logits)
    exp_logits = Nx.exp(Nx.subtract(scaled_logits, max_logit))
    probs = Nx.divide(exp_logits, Nx.sum(exp_logits))

    # Top-p sampling on CPU
    probs_list = Nx.to_flat_list(probs)

    indexed_probs = probs_list
      |> Enum.with_index()
      |> Enum.sort_by(fn {prob, _idx} -> -prob end)

    {filtered_probs, filtered_indices} = apply_top_p(indexed_probs, top_p)

    # Renormalize and sample
    total = Enum.sum(filtered_probs)
    normalized_probs = Enum.map(filtered_probs, &(&1 / total))

    selected_idx = sample_from_distribution(normalized_probs, filtered_indices)
    Nx.tensor(selected_idx)
  end

  defp apply_top_p(indexed_probs, top_p) do
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
  end

  defp sample_from_distribution(probs, indices) do
    random_val = :rand.uniform()

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

  defp detect_eos_token(tokenizer) do
    # Try to get EOS token from tokenizer config
    case tokenizer do
      %{special_tokens: %{eos: %{id: id}}} -> id
      _ -> 151643  # Qwen3 default
    end
  end
end
