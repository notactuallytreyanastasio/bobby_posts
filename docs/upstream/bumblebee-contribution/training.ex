defmodule Bumblebee.Training.LoRA do
  @moduledoc """
  LoRA training integration for Bumblebee models.

  This module provides a streamlined interface for fine-tuning quantized
  models using LoRA adapters. Training happens via Python's `mlx_lm` package,
  with Elixir handling data preparation and result loading.

  ## Full Workflow Example

      # 1. Prepare your training data
      posts = [
        "First example post about Elixir",
        "Another post discussing functional programming",
        "A third post about pattern matching"
        # ... more examples
      ]

      # 2. Create training dataset
      Bumblebee.Training.LoRA.prepare_data(posts, "/path/to/data",
        prompt: "Write a post in my style",
        split_ratio: 0.9
      )

      # 3. Train the adapter
      {:ok, adapter_path} = Bumblebee.Training.LoRA.train(
        base_model: "lmstudio-community/Qwen3-8B-MLX-4bit",
        training_data: "/path/to/data",
        output_path: "/path/to/my-adapter",
        iterations: 25_000,
        rank: 8,
        scale: 20.0
      )

      # 4. Load and use the adapter
      {:ok, model} = Bumblebee.QuantizedLoader.load_model("/path/to/Qwen3-8B")
      {:ok, adapter} = Bumblebee.Adapters.load(adapter_path)
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-8B"})

      serving = Bumblebee.Text.QuantizedGeneration.new(model, tokenizer,
        adapter: adapter
      )

      Nx.Serving.run(serving, "Write a post")

  ## Training Data Format

  The training data is stored in ChatML JSONL format:

      {"messages": [{"role": "user", "content": "Write a post"}, {"role": "assistant", "content": "Your content here"}]}

  This format is compatible with most instruction-tuned models including
  Qwen3, LLaMA, Mistral, etc.

  ## Requirements

  - Python 3.10+
  - `mlx_lm` package: `pip install mlx_lm`
  - Apple Silicon Mac (M1/M2/M3/M4)
  - Sufficient disk space for checkpoints (~100MB per save)

  ## Hyperparameter Guidelines

  | Use Case | Iterations | Rank | Scale | Learning Rate |
  |----------|------------|------|-------|---------------|
  | Quick test | 1,000 | 4 | 16.0 | 1e-5 |
  | Light fine-tune | 5,000 | 8 | 20.0 | 1e-5 |
  | Full training | 25,000 | 8 | 20.0 | 1e-5 |
  | Heavy adaptation | 50,000 | 16 | 24.0 | 5e-6 |

  For style transfer (like learning someone's writing style), we recommend:
  - 25,000 iterations
  - rank=8, scale=20.0
  - Filter training data to desired length (e.g., >160 chars)
  - Use `mask_prompt: true` to only learn on completions
  """

  require Logger

  @doc """
  Prepares training data in ChatML JSONL format.

  ## Parameters

    * `contents` - List of content strings to train on
    * `output_dir` - Directory to write training files
    * `opts` - Options:
      * `:prompt` - System prompt for all examples (default: "Write a post")
      * `:split_ratio` - Train/validation split (default: 0.9)
      * `:min_length` - Minimum content length (default: nil)
      * `:shuffle_seed` - Random seed for shuffling (default: nil, random)

  ## Returns

  `:ok` on success, `{:error, reason}` on failure.

  Creates two files:
  - `{output_dir}/train.jsonl`
  - `{output_dir}/valid.jsonl`
  """
  @spec prepare_data([String.t()], Path.t(), keyword()) :: :ok | {:error, term()}
  def prepare_data(contents, output_dir, opts \\ []) do
    prompt = Keyword.get(opts, :prompt, "Write a post")
    split_ratio = Keyword.get(opts, :split_ratio, 0.9)
    min_length = Keyword.get(opts, :min_length)
    shuffle_seed = Keyword.get(opts, :shuffle_seed)

    # Filter by minimum length if specified
    contents = if min_length do
      Enum.filter(contents, &(String.length(&1) >= min_length))
    else
      contents
    end

    if Enum.empty?(contents) do
      {:error, :no_valid_content}
    else
      File.mkdir_p!(output_dir)

      # Convert to ChatML format
      messages = Enum.map(contents, fn content ->
        %{
          "messages" => [
            %{"role" => "user", "content" => prompt},
            %{"role" => "assistant", "content" => content}
          ]
        }
      end)

      # Shuffle with optional seed
      if shuffle_seed, do: :rand.seed(:exsss, {shuffle_seed, shuffle_seed, shuffle_seed})
      shuffled = Enum.shuffle(messages)

      # Split
      split_idx = round(length(shuffled) * split_ratio)
      {train, valid} = Enum.split(shuffled, split_idx)

      # Write files
      write_jsonl(Path.join(output_dir, "train.jsonl"), train)
      write_jsonl(Path.join(output_dir, "valid.jsonl"), valid)

      Logger.info("[Bumblebee.Training.LoRA] Created training data:")
      Logger.info("  Train: #{length(train)} samples")
      Logger.info("  Valid: #{length(valid)} samples")
      Logger.info("  Average length: #{avg_length(contents)} chars")

      :ok
    end
  end

  @doc """
  Trains a LoRA adapter using mlx_lm.

  ## Parameters

    * `opts` - Training options (all paths are required):
      * `:base_model` - HuggingFace model ID or local path
      * `:training_data` - Directory with train.jsonl and valid.jsonl
      * `:output_path` - Where to save the trained adapter
      * `:iterations` - Number of training iterations (default: 1000)
      * `:rank` - LoRA rank (default: 8)
      * `:scale` - LoRA scale (default: 20.0)
      * `:learning_rate` - Learning rate (default: 1e-5)
      * `:batch_size` - Batch size (default: 1)
      * `:grad_accumulation` - Gradient accumulation steps (default: 8)
      * `:max_seq_length` - Maximum sequence length (default: 256)
      * `:save_every` - Save checkpoint interval (default: iterations/10)
      * `:num_layers` - Number of layers to fine-tune (default: all)

  ## Returns

  `{:ok, output_path}` on successful training, `{:error, reason}` on failure.
  """
  @spec train(keyword()) :: {:ok, Path.t()} | {:error, term()}
  def train(opts) do
    with :ok <- validate_python_env(),
         :ok <- validate_required_opts(opts, [:base_model, :training_data, :output_path]),
         {:ok, config_path} <- write_training_config(opts) do

      Logger.info("[Bumblebee.Training.LoRA] Starting training...")
      Logger.info("[Bumblebee.Training.LoRA] Base model: #{opts[:base_model]}")
      Logger.info("[Bumblebee.Training.LoRA] Training data: #{opts[:training_data]}")
      Logger.info("[Bumblebee.Training.LoRA] Output: #{opts[:output_path]}")

      case run_training(config_path) do
        :ok ->
          File.rm(config_path)
          {:ok, opts[:output_path]}

        {:error, reason} ->
          File.rm(config_path)
          {:error, reason}
      end
    end
  end

  @doc """
  Monitors training progress from a log file.

  Returns a stream of training metrics that can be consumed in real-time.
  """
  @spec monitor_training(Path.t()) :: Enumerable.t()
  def monitor_training(log_file) do
    Stream.resource(
      fn -> File.open!(log_file, [:read]) end,
      fn file ->
        case IO.read(file, :line) do
          :eof ->
            Process.sleep(1000)  # Wait for more output
            case IO.read(file, :line) do
              :eof -> {[], file}
              line -> {[parse_log_line(line)], file}
            end
          line ->
            {[parse_log_line(line)], file}
        end
      end,
      fn file -> File.close(file) end
    )
    |> Stream.reject(&is_nil/1)
  end

  @doc """
  Estimates training time based on iterations and model size.
  """
  @spec estimate_training_time(pos_integer(), atom()) :: String.t()
  def estimate_training_time(iterations, model_size \\ :qwen3_8b) do
    # Rough estimates based on M2 Pro
    secs_per_iter = case model_size do
      :qwen3_8b -> 0.8
      :llama_7b -> 0.6
      :mistral_7b -> 0.7
      _ -> 1.0
    end

    total_secs = iterations * secs_per_iter
    hours = div(trunc(total_secs), 3600)
    mins = div(rem(trunc(total_secs), 3600), 60)

    "~#{hours}h #{mins}m"
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp validate_python_env do
    case System.cmd("python", ["-c", "import mlx_lm"], stderr_to_stdout: true) do
      {_, 0} -> :ok
      {output, _} ->
        Logger.error("[Bumblebee.Training.LoRA] mlx_lm not found. Install with: pip install mlx_lm")
        {:error, {:mlx_lm_not_found, output}}
    end
  end

  defp validate_required_opts(opts, required) do
    missing = Enum.filter(required, fn key -> not Keyword.has_key?(opts, key) end)
    if Enum.empty?(missing) do
      :ok
    else
      {:error, {:missing_options, missing}}
    end
  end

  defp write_training_config(opts) do
    iterations = Keyword.get(opts, :iterations, 1000)

    config = %{
      "model" => opts[:base_model],
      "data" => opts[:training_data],
      "adapter_path" => opts[:output_path],
      "train" => true,
      "fine_tune_type" => "lora",
      "batch_size" => Keyword.get(opts, :batch_size, 1),
      "grad_accumulation_steps" => Keyword.get(opts, :grad_accumulation, 8),
      "iters" => iterations,
      "learning_rate" => Keyword.get(opts, :learning_rate, 1.0e-5),
      "max_seq_length" => Keyword.get(opts, :max_seq_length, 256),
      "save_every" => Keyword.get(opts, :save_every, div(iterations, 10)),
      "steps_per_report" => Keyword.get(opts, :steps_per_report, 100),
      "steps_per_eval" => Keyword.get(opts, :steps_per_eval, div(iterations, 10)),
      "mask_prompt" => true,
      "seed" => Keyword.get(opts, :seed, 42),
      "lora_parameters" => %{
        "rank" => Keyword.get(opts, :rank, 8),
        "dropout" => Keyword.get(opts, :dropout, 0.0),
        "scale" => Keyword.get(opts, :scale, 20.0)
      }
    }

    # Add num_layers if specified
    config = if num_layers = opts[:num_layers] do
      Map.put(config, "num_layers", num_layers)
    else
      config
    end

    config_path = Path.join(System.tmp_dir!(), "lora_config_#{:rand.uniform(100_000)}.yaml")

    # Write as YAML-like format (JSON works too for mlx_lm)
    case File.write(config_path, Jason.encode!(config, pretty: true)) do
      :ok -> {:ok, config_path}
      {:error, reason} -> {:error, {:config_write_failed, reason}}
    end
  end

  defp run_training(config_path) do
    case System.cmd("python", ["-m", "mlx_lm.lora", "--config", config_path],
           stderr_to_stdout: true,
           into: IO.stream(:stdio, :line)
         ) do
      {_, 0} ->
        Logger.info("[Bumblebee.Training.LoRA] Training complete!")
        :ok

      {output, exit_code} ->
        Logger.error("[Bumblebee.Training.LoRA] Training failed with exit code #{exit_code}")
        {:error, {:training_failed, exit_code, output}}
    end
  end

  defp write_jsonl(path, items) do
    content = items
      |> Enum.map(&Jason.encode!/1)
      |> Enum.join("\n")

    File.write!(path, content <> "\n")
  end

  defp avg_length(contents) do
    total = Enum.reduce(contents, 0, fn c, acc -> acc + String.length(c) end)
    div(total, length(contents))
  end

  defp parse_log_line(line) do
    cond do
      String.contains?(line, "Train Loss:") ->
        case Regex.run(~r/Iter (\d+): Train Loss: ([\d.]+)/, line) do
          [_, iter, loss] -> %{type: :train, iter: String.to_integer(iter), loss: String.to_float(loss)}
          _ -> nil
        end

      String.contains?(line, "Val Loss:") ->
        case Regex.run(~r/Val Loss: ([\d.]+)/, line) do
          [_, loss] -> %{type: :validation, loss: String.to_float(loss)}
          _ -> nil
        end

      true -> nil
    end
  end
end
