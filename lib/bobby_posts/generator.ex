defmodule BobbyPosts.Generator do
  @moduledoc """
  GenServer that holds the loaded Qwen3 model and provides text generation.

  The model is lazy-loaded on first generation request to avoid slow startup.
  Uses base model + LoRA adapters for generation (not fused model).

  Pure Elixir - no Python subprocess calls.
  """

  use GenServer
  require Logger

  alias BobbyPosts.{QuantizedLoader, AdapterLoader, Tokenizer, Qwen3.Generate}

  # Base model path (not fused - adapters applied at runtime)
  @default_model_path "/Users/robertgrayson/.cache/huggingface/hub/models--lmstudio-community--Qwen3-8B-MLX-4bit/snapshots/a84107f5c4dfdecf389b208598faeac322048237"
  @default_adapter_path "/Users/robertgrayson/twitter_finetune/bobby_posts_adapters/adapters/v5"
  @eos_token_id 151645  # Qwen3 <|im_end|> token
  @bluesky_char_limit 300
  @min_char_length 80  # Retry if post is shorter than this (lowered - model trained on tweets)

  # Default prompt - explicitly ask for longer content
  @default_prompt "Write a post that is at least 200 characters long in your authentic voice. Expand on your thought, add context or a second sentence. Be chaotic but personable and compassionate and creative."

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
    adapter_path = Keyword.get(opts, :adapter_path, get_adapter_path())

    {:ok, %{
      model: nil,
      adapters: nil,
      tokenizer: nil,
      model_path: model_path,
      adapter_path: adapter_path,
      loading: false
    }}
  end

  @impl true
  def handle_call(:loaded?, _from, state) do
    {:reply, state.model != nil, state}
  end

  @impl true
  def handle_call(:load_model, _from, %{model: nil} = state) do
    case do_load_all(state.model_path, state.adapter_path) do
      {:ok, model, adapters, tokenizer} ->
        {:reply, :ok, %{state | model: model, adapters: adapters, tokenizer: tokenizer}}
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
    case do_load_all(state.model_path, state.adapter_path) do
      {:ok, model, adapters, tokenizer} ->
        state = %{state | model: model, adapters: adapters, tokenizer: tokenizer}
        result = do_generate(state, opts)
        {:reply, result, state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:generate, opts}, _from, state) do
    result = do_generate(state, opts)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:trace_generate, opts}, _from, %{model: nil} = state) do
    Logger.info("Model not loaded, loading now...")
    case do_load_all(state.model_path, state.adapter_path) do
      {:ok, model, adapters, tokenizer} ->
        state = %{state | model: model, adapters: adapters, tokenizer: tokenizer}
        result = do_trace_generate(state, opts)
        {:reply, result, state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:trace_generate, opts}, _from, state) do
    result = do_trace_generate(state, opts)
    {:reply, result, state}
  end

  # Private Functions

  defp get_model_path do
    Application.get_env(:bobby_posts, :model_path, @default_model_path)
  end

  defp get_adapter_path do
    Application.get_env(:bobby_posts, :adapter_path, @default_adapter_path)
  end

  defp do_load_all(model_path, adapter_path) do
    Logger.info("Loading model, adapters, and tokenizer...")
    start_time = System.monotonic_time(:millisecond)

    with {:ok, tokenizer} <- Tokenizer.load(model_path),
         _ = Logger.info("Tokenizer loaded"),
         {:ok, model} <- QuantizedLoader.load_model(model_path),
         _ = Logger.info("Base model loaded"),
         {:ok, adapters} <- load_adapters_if_present(adapter_path) do
      elapsed = System.monotonic_time(:millisecond) - start_time
      Logger.info("All components loaded in #{elapsed}ms")
      {:ok, model, adapters, tokenizer}
    else
      {:error, reason} ->
        Logger.error("Failed to load: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp load_adapters_if_present(nil), do: {:ok, nil}
  defp load_adapters_if_present(adapter_path) do
    Logger.info("Loading adapters from #{adapter_path}...")
    AdapterLoader.load_adapters(adapter_path)
  end

  defp do_generate(state, opts) do
    prompt = Keyword.get(opts, :prompt)
    max_tokens = Keyword.get(opts, :max_tokens, 280)
    count = Keyword.get(opts, :count, 1)
    temperature = Keyword.get(opts, :temperature, 0.95)
    top_p = Keyword.get(opts, :top_p, 0.9)
    char_limit = Keyword.get(opts, :char_limit, @bluesky_char_limit)

    posts =
      for _i <- 1..count do
        generate_single(state, prompt, max_tokens, temperature, top_p, char_limit)
      end

    {:ok, posts}
  end

  defp generate_single(state, prompt, max_tokens, temperature, top_p, char_limit) do
    generate_with_retry(state, prompt, max_tokens, temperature, top_p, char_limit, 5)
  end

  defp generate_with_retry(_state, _prompt, _max_tokens, _temperature, _top_p, _char_limit, 0) do
    "(generation failed - please try again)"
  end

  defp generate_with_retry(state, prompt, max_tokens, temperature, top_p, char_limit, retries) do
    %{model: model, adapters: adapters, tokenizer: tokenizer} = state

    # Build ChatML prompt
    chatml_prompt = build_chatml_prompt(prompt)

    # Tokenize using pure Elixir (Bumblebee)
    tokens = Tokenizer.encode(tokenizer, chatml_prompt)

    # Generate with adapters
    input_ids = Nx.tensor([tokens], type: :s32)

    generated_tokens = Generate.generate(input_ids, model,
      max_tokens: max_tokens,
      eos_token_id: @eos_token_id,
      temperature: temperature,
      top_p: top_p,
      adapters: adapters
    )

    # Decode using pure Elixir (Bumblebee)
    all_tokens = tokens ++ generated_tokens
    text = Tokenizer.decode(tokenizer, all_tokens)

    # Clean up the response and enforce character limit
    result = text
      |> clean_response()
      |> enforce_char_limit(char_limit)

    # Retry if empty or too short
    trimmed = String.trim(result)
    cond do
      trimmed == "" ->
        Logger.debug("Empty result, retrying (#{retries - 1} left)")
        generate_with_retry(state, prompt, max_tokens, temperature, top_p, char_limit, retries - 1)

      String.length(trimmed) < @min_char_length and retries > 1 ->
        Logger.debug("Post too short (#{String.length(trimmed)} chars), retrying (#{retries - 1} left)")
        generate_with_retry(state, prompt, max_tokens, temperature, top_p, char_limit, retries - 1)

      true ->
        result
    end
  end

  defp build_chatml_prompt(nil) do
    # Default prompt with /no_think to disable Qwen3's chain-of-thought
    "<|im_start|>user\n#{@default_prompt} /no_think<|im_end|>\n<|im_start|>assistant\n"
  end

  defp build_chatml_prompt(prompt) do
    # Add /no_think to disable thinking mode for faster, direct responses
    "<|im_start|>user\n#{prompt} /no_think<|im_end|>\n<|im_start|>assistant\n"
  end

  defp clean_response(text) do
    text
    # Extract assistant response
    |> extract_assistant_response()
    # Remove thinking blocks
    |> remove_thinking_blocks()
    # Strip hashtags
    |> strip_hashtags()
    # Strip emojis
    |> strip_emojis()
    # Clean up whitespace
    |> String.trim()
  end

  defp extract_assistant_response(text) do
    # Handle both special token format and plain text format
    # Bumblebee may decode special tokens as plain text
    text
    |> try_extract_with_markers()
    |> try_extract_plain_assistant()
  end

  defp try_extract_with_markers(text) do
    case String.split(text, "<|im_start|>assistant\n", parts: 2) do
      [_, response] ->
        case String.split(response, "<|im_end|>", parts: 2) do
          [content, _] -> content
          [content] -> content
        end
      _ -> text
    end
  end

  defp try_extract_plain_assistant(text) do
    # Handle case where Bumblebee decodes as plain "assistant" text
    # Pattern: "...user ... /no_think assistant ACTUAL_CONTENT"
    case Regex.run(~r/assistant\s+(.+)$/s, text) do
      [_, content] -> String.trim(content)
      _ -> text
    end
  end

  defp remove_thinking_blocks(text) do
    # Remove <think>...</think> blocks (Qwen3's reasoning mode)
    # Also handle unclosed <think> blocks
    text
    |> then(&Regex.replace(~r/<think>.*?<\/think>/s, &1, ""))
    |> then(&Regex.replace(~r/<think>.*$/s, &1, ""))
  end

  defp strip_hashtags(text) do
    # Remove hashtags - user never uses them
    text
    |> String.replace(~r/#\w+\s*/u, "")
    |> String.replace(~r/\s+/, " ")
  end

  defp strip_emojis(text) do
    # Remove emojis - comprehensive Unicode emoji ranges
    text
    |> String.replace(~r/[\x{1F600}-\x{1F64F}]/u, "")  # Emoticons
    |> String.replace(~r/[\x{1F300}-\x{1F5FF}]/u, "")  # Misc symbols & pictographs
    |> String.replace(~r/[\x{1F680}-\x{1F6FF}]/u, "")  # Transport & map symbols
    |> String.replace(~r/[\x{1F1E0}-\x{1F1FF}]/u, "")  # Flags
    |> String.replace(~r/[\x{2600}-\x{26FF}]/u, "")    # Misc symbols
    |> String.replace(~r/[\x{2700}-\x{27BF}]/u, "")    # Dingbats
    |> String.replace(~r/[\x{FE00}-\x{FE0F}]/u, "")    # Variation selectors
    |> String.replace(~r/[\x{1F900}-\x{1F9FF}]/u, "")  # Supplemental symbols
    |> String.replace(~r/[\x{1FA00}-\x{1FA6F}]/u, "")  # Chess, extended-A
    |> String.replace(~r/[\x{1FA70}-\x{1FAFF}]/u, "")  # Symbols extended-A
    |> String.replace(~r/[\x{231A}-\x{231B}]/u, "")    # Watch, hourglass
    |> String.replace(~r/[\x{23E9}-\x{23F3}]/u, "")    # Media controls
    |> String.replace(~r/[\x{23F8}-\x{23FA}]/u, "")    # More media
    |> String.replace(~r/[\x{25AA}-\x{25AB}]/u, "")    # Squares
    |> String.replace(~r/[\x{25B6}]/u, "")             # Play button
    |> String.replace(~r/[\x{25C0}]/u, "")             # Reverse button
    |> String.replace(~r/[\x{25FB}-\x{25FE}]/u, "")    # More squares
    |> String.replace(~r/[\x{2614}-\x{2615}]/u, "")    # Umbrella, hot beverage
    |> String.replace(~r/[\x{2648}-\x{2653}]/u, "")    # Zodiac
    |> String.replace(~r/[\x{267F}]/u, "")             # Wheelchair
    |> String.replace(~r/[\x{2693}]/u, "")             # Anchor
    |> String.replace(~r/[\x{26A1}]/u, "")             # High voltage
    |> String.replace(~r/[\x{26AA}-\x{26AB}]/u, "")    # Circles
    |> String.replace(~r/[\x{26BD}-\x{26BE}]/u, "")    # Sports
    |> String.replace(~r/[\x{26C4}-\x{26C5}]/u, "")    # Weather
    |> String.replace(~r/[\x{26CE}]/u, "")             # Ophiuchus
    |> String.replace(~r/[\x{26D4}]/u, "")             # No entry
    |> String.replace(~r/[\x{26EA}]/u, "")             # Church
    |> String.replace(~r/[\x{26F2}-\x{26F3}]/u, "")    # Fountain, golf
    |> String.replace(~r/[\x{26F5}]/u, "")             # Sailboat
    |> String.replace(~r/[\x{26FA}]/u, "")             # Tent
    |> String.replace(~r/[\x{26FD}]/u, "")             # Fuel pump
    |> String.replace(~r/[\x{2702}]/u, "")             # Scissors
    |> String.replace(~r/[\x{2705}]/u, "")             # Check mark
    |> String.replace(~r/[\x{2708}-\x{270D}]/u, "")    # Airplane to writing hand
    |> String.replace(~r/[\x{270F}]/u, "")             # Pencil
    |> String.replace(~r/[\x{2712}]/u, "")             # Black nib
    |> String.replace(~r/[\x{2714}]/u, "")             # Check mark
    |> String.replace(~r/[\x{2716}]/u, "")             # X mark
    |> String.replace(~r/[\x{271D}]/u, "")             # Latin cross
    |> String.replace(~r/[\x{2721}]/u, "")             # Star of David
    |> String.replace(~r/[\x{2728}]/u, "")             # Sparkles
    |> String.replace(~r/[\x{2733}-\x{2734}]/u, "")    # Eight spoked asterisk
    |> String.replace(~r/[\x{2744}]/u, "")             # Snowflake
    |> String.replace(~r/[\x{2747}]/u, "")             # Sparkle
    |> String.replace(~r/[\x{274C}]/u, "")             # Cross mark
    |> String.replace(~r/[\x{274E}]/u, "")             # Cross mark
    |> String.replace(~r/[\x{2753}-\x{2755}]/u, "")    # Question marks
    |> String.replace(~r/[\x{2757}]/u, "")             # Exclamation
    |> String.replace(~r/[\x{2763}-\x{2764}]/u, "")    # Heart exclamation, heart
    |> String.replace(~r/[\x{2795}-\x{2797}]/u, "")    # Math symbols
    |> String.replace(~r/[\x{27A1}]/u, "")             # Right arrow
    |> String.replace(~r/[\x{27B0}]/u, "")             # Curly loop
    |> String.replace(~r/[\x{27BF}]/u, "")             # Double curly loop
    |> String.replace(~r/[\x{2934}-\x{2935}]/u, "")    # Arrows
    |> String.replace(~r/[\x{2B05}-\x{2B07}]/u, "")    # Arrows
    |> String.replace(~r/[\x{2B1B}-\x{2B1C}]/u, "")    # Squares
    |> String.replace(~r/[\x{2B50}]/u, "")             # Star
    |> String.replace(~r/[\x{2B55}]/u, "")             # Circle
    |> String.replace(~r/[\x{3030}]/u, "")             # Wavy dash
    |> String.replace(~r/[\x{303D}]/u, "")             # Part alternation mark
    |> String.replace(~r/[\x{3297}]/u, "")             # Circled ideograph congratulation
    |> String.replace(~r/[\x{3299}]/u, "")             # Circled ideograph secret
    |> String.replace(~r/\s+/, " ")                    # Clean up multiple spaces
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

  defp do_trace_generate(state, opts) do
    %{model: model, adapters: adapters, tokenizer: tokenizer} = state
    prompt = Keyword.get(opts, :prompt)
    max_tokens = Keyword.get(opts, :max_tokens, 10)

    # Build ChatML prompt
    chatml_prompt = build_chatml_prompt(prompt)
    Logger.info("ChatML prompt: #{inspect(chatml_prompt)}")

    # Tokenize using pure Elixir
    tokens = Tokenizer.encode(tokenizer, chatml_prompt)
    Logger.info("Input tokens (#{length(tokens)}): #{inspect(tokens)}")

    # Generate with adapters
    input_ids = Nx.tensor([tokens], type: :s32)

    generated_tokens = Generate.generate(input_ids, model,
      max_tokens: max_tokens,
      eos_token_id: @eos_token_id,
      adapters: adapters
    )

    Logger.info("Generated tokens: #{inspect(generated_tokens)}")

    # Decode each token individually using pure Elixir
    for {token, i} <- Enum.with_index(generated_tokens) do
      text = Tokenizer.decode(tokenizer, [token])
      Logger.info("Token #{i}: #{token} -> #{inspect(text)}")
    end

    {:ok, generated_tokens}
  end
end
