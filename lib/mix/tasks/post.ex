defmodule Mix.Tasks.Post do
  @moduledoc """
  Generates posts in Bobby's voice using the fine-tuned Qwen3 model.

  ## Usage

      mix post                           # Generate 1 post with default prompt
      mix post 5                         # Generate 5 posts
      mix post "your prompt here"        # Generate with custom prompt
      mix post 3 "your prompt"           # Generate 3 posts with prompt
      mix post --count 3                 # Generate 3 posts (flag style)

  ## Options

    * `-c, --count` - Number of posts to generate (default: 1)
    * `-m, --max-tokens` - Maximum tokens per generation (default: 100)
    * `-v, --verbose` - Enable verbose logging

  ## Examples

      mix post
      mix post 4
      mix post "What's your take on Elixir?"
      mix post 3 "Hot take on AI"
      mix post --count 5 --max-tokens 100

  """

  use Mix.Task
  require Logger

  @shortdoc "Generate posts in Bobby's voice"

  @switches [
    count: :integer,
    max_tokens: :integer,
    verbose: :boolean
  ]

  @aliases [
    c: :count,
    m: :max_tokens,
    v: :verbose
  ]

  # Parse count from first positional arg if it's a number
  defp parse_count_from_args([first | rest], nil) do
    case Integer.parse(first) do
      {n, ""} when n > 0 -> {n, rest}
      _ -> {1, [first | rest]}
    end
  end
  defp parse_count_from_args(rest, nil), do: {1, rest}
  defp parse_count_from_args(rest, count), do: {count, rest}

  @impl Mix.Task
  def run(args) do
    # Start the application to get all dependencies
    Application.ensure_all_started(:bobby_posts)

    {opts, rest, _} = OptionParser.parse(args, switches: @switches, aliases: @aliases)

    # Check if first positional arg is a number (count)
    {count, rest} = parse_count_from_args(rest, Keyword.get(opts, :count))
    max_tokens = Keyword.get(opts, :max_tokens, 300)
    verbose = Keyword.get(opts, :verbose, false)

    # Join remaining args as prompt
    prompt = case rest do
      [] -> nil
      parts -> Enum.join(parts, " ")
    end

    if verbose do
      Logger.configure(level: :debug)
    end

    # Generator is already started by the application supervisor

    IO.puts("\n🤖 Bobby Posts Generator")
    IO.puts("========================\n")

    if prompt do
      IO.puts("Prompt: #{prompt}")
    else
      IO.puts("Prompt: (default - open-ended)")
    end

    IO.puts("Count: #{count}")
    IO.puts("Max tokens: #{max_tokens}")
    IO.puts("\nLoading model and generating...\n")

    start_time = System.monotonic_time(:millisecond)

    case BobbyPosts.Generator.generate(prompt: prompt, max_tokens: max_tokens, count: count) do
      {:ok, posts} ->
        elapsed = System.monotonic_time(:millisecond) - start_time

        Enum.with_index(posts, 1)
        |> Enum.each(fn {post, idx} ->
          IO.puts("--- Post #{idx} ---")
          IO.puts(post)
          IO.puts("")
        end)

        IO.puts("Generated #{count} post(s) in #{elapsed}ms")

      {:error, reason} ->
        IO.puts("❌ Error: #{inspect(reason)}")
        System.halt(1)
    end
  end
end
