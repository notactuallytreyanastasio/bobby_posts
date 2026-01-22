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
    max_tokens = Keyword.get(opts, :max_tokens, 200)
    count = Keyword.get(opts, :count, 1)

    posts =
      for _i <- 1..count do
        generate_single(model, prompt, max_tokens)
      end

    {:ok, posts}
  end

  defp generate_single(model, prompt, max_tokens) do
    # Build ChatML prompt
    chatml_prompt = build_chatml_prompt(prompt)

    # Tokenize using Python subprocess
    tokens = tokenize(chatml_prompt)

    # Generate
    input_ids = Nx.tensor([tokens], type: :s32)

    generated_tokens = Generate.generate(input_ids, model,
      max_tokens: max_tokens,
      eos_token_id: @eos_token_id
    )

    # Decode
    all_tokens = tokens ++ generated_tokens
    text = decode(all_tokens)

    # Clean up the response
    clean_response(text)
  end

  defp build_chatml_prompt(nil) do
    # Default prompt for authentic chaotic Bluesky voice
    "<|im_start|>user\nmake a bluesky post in your authentic voice. skew towards chaotic<|im_end|>\n<|im_start|>assistant\n"
  end

  defp build_chatml_prompt(prompt) do
    "<|im_start|>user\n#{prompt}<|im_end|>\n<|im_start|>assistant\n"
  end

  defp tokenize(text) do
    # Use Python subprocess for tokenization (HuggingFace transformers)
    model_path = get_model_path()

    python_code = """
import sys
sys.path.insert(0, '.')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('#{model_path}', trust_remote_code=True)
tokens = tokenizer.encode('#{escape_for_python(text)}', add_special_tokens=False)
print(','.join(map(str, tokens)))
"""

    case System.cmd("python3", ["-c", python_code], stderr_to_stdout: true) do
      {output, 0} ->
        output
        |> String.trim()
        |> String.split(",")
        |> Enum.map(&String.to_integer/1)

      {error, _} ->
        Logger.error("Tokenization failed: #{error}")
        []
    end
  end

  defp decode(tokens) do
    model_path = get_model_path()
    token_str = Enum.join(tokens, ",")

    python_code = """
import sys
sys.path.insert(0, '.')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('#{model_path}', trust_remote_code=True)
tokens = [#{token_str}]
text = tokenizer.decode(tokens, skip_special_tokens=False)
print(text)
"""

    case System.cmd("python3", ["-c", python_code], stderr_to_stdout: true) do
      {output, 0} ->
        String.trim(output)

      {error, _} ->
        Logger.error("Decoding failed: #{error}")
        ""
    end
  end

  defp escape_for_python(text) do
    text
    |> String.replace("\\", "\\\\")
    |> String.replace("'", "\\'")
    |> String.replace("\n", "\\n")
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
end
