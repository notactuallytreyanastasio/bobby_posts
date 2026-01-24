defmodule BobbyPosts.Tokenizer do
  @moduledoc """
  Pure Elixir tokenizer for Qwen3 using Bumblebee.

  No Python. No subprocess. Just Elixir.
  """

  require Logger

  @doc """
  Loads the tokenizer from a local model path.

  Returns `{:ok, tokenizer}` or `{:error, reason}`.
  """
  @spec load(Path.t()) :: {:ok, struct()} | {:error, term()}
  def load(model_path) do
    Logger.info("Loading tokenizer from #{model_path}")
    Bumblebee.load_tokenizer({:local, model_path})
  end

  @doc """
  Encodes text into token IDs.

  Returns a list of token IDs.
  """
  @spec encode(struct(), String.t()) :: list(integer())
  def encode(tokenizer, text) do
    inputs = Bumblebee.apply_tokenizer(tokenizer, text)

    # Extract token IDs from the input tensor
    inputs["input_ids"]
    |> Nx.to_flat_list()
  end

  @doc """
  Decodes token IDs back into text.

  Returns the decoded string.
  """
  @spec decode(struct(), list(integer())) :: String.t()
  def decode(tokenizer, token_ids) when is_list(token_ids) do
    Bumblebee.Tokenizer.decode(tokenizer, token_ids)
  end

  @doc """
  Gets a special token ID by type (e.g., :eos, :pad, :bos).
  """
  @spec special_token_id(struct(), atom()) :: integer() | nil
  def special_token_id(tokenizer, type) do
    Bumblebee.Tokenizer.special_token_id(tokenizer, type)
  end
end
