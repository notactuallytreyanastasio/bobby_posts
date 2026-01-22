defmodule BobbyPosts.Generator do
  @moduledoc """
  GenServer that holds the loaded Qwen3 model and provides text generation.

  The model is lazy-loaded on first generation request to avoid slow startup.
  """

  use GenServer
  require Logger

  alias BobbyPosts.{QuantizedLoader, Qwen3.Generate}

  @default_model_path "/Users/robertgrayson/twitter_finetune/fused_model"
  @eos_token_id 151645  # Qwen3 <|im_end|> token
  @bluesky_char_limit 300

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Generates text in Bobby's voice.

  ## Options
    - `:prompt` - Optional prompt to guide generation
    - `:max_tokens` - Maximum tokens to generate (default: 200)
    - `:count` - Number of posts to generate (default: 1)
  """
  def generate(opts \\ []) do
    GenServer.call(__MODULE__, {:generate, opts}, :infinity)
  end

  @doc """
  Checks if the model is loaded.
  """
  def loaded? do
    GenServer.call(__MODULE__, :loaded?)
  end

  @doc """
  Forces model loading (useful for prewarming).
  """
  def load_model do
    GenServer.call(__MODULE__, :load_model, :infinity)
  end

  @doc """
  Gets model memory statistics.
  """
  def memory_stats do
    GenServer.call(__MODULE__, :memory_stats)
  end

  @doc """
  Traces token generation for debugging.
  """
  def trace_generate(opts \\ []) do
    GenServer.call(__MODULE__, {:trace_generate, opts}, :infinity)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    model_path = Keyword.get(opts, :model_path, get_model_path())

    {:ok, %{
      model: nil,
      model_path: model_path,
      loading: false
    }}
  end

  @impl true
  def handle_call(:loaded?, _from, state) do
    {:reply, state.model != nil, state}
  end

  @impl true
  def handle_call(:load_model, _from, %{model: nil} = state) do
    case do_load_model(state.model_path) do
      {:ok, model} ->
        {:reply, :ok, %{state | model: model}}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call(:load_model, _from, state) do
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:memory_stats, _from, %{model: nil} = state) do
    {:reply, {:error, :model_not_loaded}, state}
  end

  def handle_call(:memory_stats, _from, %{model: model} = state) do
    stats = QuantizedLoader.memory_stats(model)
    {:reply, {:ok, stats}, state}
  end

  @impl true
  def handle_call({:generate, opts}, _from, %{model: nil} = state) do
    Logger.info("Model not loaded, loading now...")
    case do_load_model(state.model_path) do
      {:ok, model} ->
        state = %{state | model: model}
        result = do_generate(model, opts)
        {:reply, result, state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:generate, opts}, _from, %{model: model} = state) do
    result = do_generate(model, opts)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:trace_generate, opts}, _from, %{model: nil} = state) do
    Logger.info("Model not loaded, loading now...")
    case do_load_model(state.model_path) do
      {:ok, model} ->
        state = %{state | model: model}
        result = do_trace_generate(model, opts)
        {:reply, result, state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:trace_generate, opts}, _from, %{model: model} = state) do
    result = do_trace_generate(model, opts)
    {:reply, result, state}
  end

  # Private Functions

  defp get_model_path do
    Application.get_env(:bobby_posts, :model_path, @default_model_path)
  end

  defp do_load_model(model_path) do
    Logger.info("Loading model from #{model_path}...")
    start_time = System.monotonic_time(:millisecond)

    case QuantizedLoader.load_model(model_path) do
      {:ok, model} ->
        elapsed = System.monotonic_time(:millisecond) - start_time
        Logger.info("Model loaded in #{elapsed}ms")
        {:ok, model}
      {:error, reason} ->
        Logger.error("Failed to load model: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp do_generate(model, opts) do
    prompt = Keyword.get(opts, :prompt)
    max_tokens = Keyword.get(opts, :max_tokens, 100)
    count = Keyword.get(opts, :count, 1)
    temperature = Keyword.get(opts, :temperature, 0.7)
    top_p = Keyword.get(opts, :top_p, 0.9)
    char_limit = Keyword.get(opts, :char_limit, @bluesky_char_limit)

    posts =
      for _i <- 1..count do
        generate_single(model, prompt, max_tokens, temperature, top_p, char_limit)
      end

    {:ok, posts}
  end

  defp generate_single(model, prompt, max_tokens, temperature, top_p, char_limit) do
    # Build ChatML prompt
    chatml_prompt = build_chatml_prompt(prompt)

    # Tokenize using Python subprocess
    tokens = tokenize(chatml_prompt)

    # Generate
    input_ids = Nx.tensor([tokens], type: :s32)

    generated_tokens = Generate.generate(input_ids, model,
      max_tokens: max_tokens,
      eos_token_id: @eos_token_id,
      temperature: temperature,
      top_p: top_p
    )

    # Decode
    all_tokens = tokens ++ generated_tokens
    text = decode(all_tokens)

    # Clean up the response and enforce character limit
    text
    |> clean_response()
    |> enforce_char_limit(char_limit)
  end

  defp build_chatml_prompt(nil) do
    # Default prompt for authentic chaotic Bluesky voice - emphasize brevity
    "<|im_start|>user\nmake a short bluesky post in your authentic voice. keep it under 280 characters. skew towards chaotic<|im_end|>\n<|im_start|>assistant\n"
  end

  defp build_chatml_prompt(prompt) do
    "<|im_start|>user\n#{prompt}<|im_end|>\n<|im_start|>assistant\n"
  end

  defp tokenize(text) do
    # Use Python subprocess for tokenization (HuggingFace transformers)
    # Write text to temp file to avoid escaping issues with special characters
    model_path = get_model_path()
    tmp_path = "/tmp/bobby_posts_tokenize_#{:rand.uniform(1_000_000)}.txt"

    File.write!(tmp_path, text)

    python_code = """
import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('#{model_path}', trust_remote_code=True)
with open('#{tmp_path}', 'r') as f:
    text = f.read()
tokens = tokenizer.encode(text, add_special_tokens=False)
print(','.join(map(str, tokens)))
"""

    result = case System.cmd("python3", ["-c", python_code], stderr_to_stdout: true) do
      {output, 0} ->
        output
        |> String.trim()
        |> String.split(",")
        |> Enum.map(&String.to_integer/1)

      {error, _} ->
        Logger.error("Tokenization failed: #{error}")
        []
    end

    File.rm(tmp_path)
    result
  end

  defp decode(tokens) do
    # Write tokens to temp file to avoid any escaping issues
    model_path = get_model_path()
    tmp_path = "/tmp/bobby_posts_decode_#{:rand.uniform(1_000_000)}.txt"
    token_str = Enum.join(tokens, ",")

    File.write!(tmp_path, token_str)

    python_code = """
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('#{model_path}', trust_remote_code=True)
with open('#{tmp_path}', 'r') as f:
    token_str = f.read()
tokens = [int(t) for t in token_str.split(',')]
text = tokenizer.decode(tokens, skip_special_tokens=False)
print(text)
"""

    result = case System.cmd("python3", ["-c", python_code], stderr_to_stdout: true) do
      {output, 0} ->
        String.trim(output)

      {error, _} ->
        Logger.error("Decoding failed: #{error}")
        ""
    end

    File.rm(tmp_path)
    result
  end

  defp clean_response(text) do
    text
    # Extract assistant response
    |> extract_assistant_response()
    # Remove thinking blocks
    |> remove_thinking_blocks()
    # Clean up whitespace
    |> String.trim()
  end

  defp extract_assistant_response(text) do
    case String.split(text, "<|im_start|>assistant\n", parts: 2) do
      [_, response] ->
        case String.split(response, "<|im_end|>", parts: 2) do
          [content, _] -> content
          [content] -> content
        end
      _ -> text
    end
  end

  defp remove_thinking_blocks(text) do
    # Remove <think>...</think> blocks (Qwen3's reasoning mode)
    Regex.replace(~r/<think>.*?<\/think>/s, text, "")
  end

  defp enforce_char_limit(text, limit) when byte_size(text) <= limit do
    text
  end

  defp enforce_char_limit(text, limit) do
    # Try to truncate at sentence boundary
    truncated = String.slice(text, 0, limit)

    # Find last sentence-ending punctuation
    case Regex.run(~r/^(.*[.!?])\s*/s, truncated) do
      [_, sentence] when byte_size(sentence) >= 50 ->
        String.trim(sentence)

      _ ->
        # No good sentence break, truncate at last space
        case String.split(truncated, ~r/\s+/) |> Enum.slice(0..-2//1) |> Enum.join(" ") do
          "" -> String.slice(text, 0, limit - 3) <> "..."
          words -> words <> "..."
        end
    end
  end

  defp do_trace_generate(model, opts) do
    prompt = Keyword.get(opts, :prompt)
    max_tokens = Keyword.get(opts, :max_tokens, 10)

    # Build ChatML prompt
    chatml_prompt = build_chatml_prompt(prompt)
    Logger.info("ChatML prompt: #{inspect(chatml_prompt)}")

    # Tokenize
    tokens = tokenize(chatml_prompt)
    Logger.info("Input tokens (#{length(tokens)}): #{inspect(tokens)}")

    # Generate
    input_ids = Nx.tensor([tokens], type: :s32)

    generated_tokens = Generate.generate(input_ids, model,
      max_tokens: max_tokens,
      eos_token_id: @eos_token_id
    )

    Logger.info("Generated tokens: #{inspect(generated_tokens)}")

    # Decode each token individually
    for {token, i} <- Enum.with_index(generated_tokens) do
      text = decode([token])
      Logger.info("Token #{i}: #{token} -> #{inspect(text)}")
    end

    {:ok, generated_tokens}
  end
end
