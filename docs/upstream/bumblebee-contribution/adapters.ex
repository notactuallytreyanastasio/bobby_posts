defmodule Bumblebee.Adapters do
  @moduledoc """
  LoRA (Low-Rank Adaptation) adapter support for Bumblebee models.

  LoRA enables efficient fine-tuning by training small adapter matrices
  instead of the full model weights. This module supports:

  - Loading pre-trained adapters from safetensors
  - Applying adapters at inference time
  - Training new adapters via mlx_lm integration

  ## How LoRA Works

  For each weight matrix W, LoRA adds a low-rank update:

      W' = W + (scale × A × B)

  Where:
  - A: [input_dim, rank] - down projection (typically rank=8)
  - B: [rank, output_dim] - up projection
  - scale: scaling factor (typically 16.0-32.0)

  During inference, instead of modifying weights, we apply:

      output = base_output + scale × (input @ A @ B)

  ## Example: Loading and Using Adapters

      # Load base model
      {:ok, model} = Bumblebee.QuantizedLoader.load_model("/path/to/Qwen3-8B-4bit")

      # Load adapter
      {:ok, adapter} = Bumblebee.Adapters.load("/path/to/my-adapter")

      # Create serving with adapter
      serving = Bumblebee.Text.QuantizedGeneration.new(model, tokenizer,
        adapter: adapter
      )

  ## Example: Training New Adapters

      # Prepare training data
      Bumblebee.Adapters.prepare_training_data(posts, "/path/to/training_data")

      # Train adapter (calls mlx_lm.lora under the hood)
      {:ok, adapter_path} = Bumblebee.Adapters.train(
        base_model: "/path/to/Qwen3-8B-4bit",
        training_data: "/path/to/training_data",
        output_path: "/path/to/my-adapter",
        iterations: 25_000,
        rank: 8,
        scale: 20.0
      )
  """

  require Logger

  @type lora_pair :: %{
          lora_a: Nx.Tensor.t(),
          lora_b: Nx.Tensor.t()
        }

  @type layer_adapters :: %{
          self_attn: %{
            q_proj: lora_pair() | nil,
            k_proj: lora_pair() | nil,
            v_proj: lora_pair() | nil,
            o_proj: lora_pair() | nil
          },
          mlp: %{
            gate_proj: lora_pair() | nil,
            up_proj: lora_pair() | nil,
            down_proj: lora_pair() | nil
          }
        }

  @type adapter :: %{
          layers: %{non_neg_integer() => layer_adapters()},
          config: map(),
          scaling: float(),
          rank: pos_integer()
        }

  # ============================================================================
  # Loading Adapters
  # ============================================================================

  @doc """
  Loads a LoRA adapter from a directory.

  The directory should contain:
  - `adapter_config.json` - Configuration with rank, scale, etc.
  - `adapters.safetensors` - The adapter weights

  ## Options

    * `:device` - Device to load to (`:gpu` or `:cpu`, default: `:gpu`)
    * `:scale_override` - Override the scale from config (optional)

  ## Returns

  `{:ok, adapter}` on success, `{:error, reason}` on failure.
  """
  @spec load(Path.t(), keyword()) :: {:ok, adapter()} | {:error, term()}
  def load(adapter_dir, opts \\ []) do
    safetensors_path = Path.join(adapter_dir, "adapters.safetensors")
    device = Keyword.get(opts, :device, :gpu)

    Logger.info("[Bumblebee.Adapters] Loading adapters from #{adapter_dir}")

    with {:ok, config} <- load_config(adapter_dir),
         {:ok, {header, _data_offset}} <- Safetensors.read_header(safetensors_path) do

      lora_params = config["lora_parameters"]
      rank = lora_params["rank"]
      scale = Keyword.get(opts, :scale_override, lora_params["scale"])

      # NOTE: mlx_lm uses scale directly, NOT scale/rank
      # This matches Python's mlx_lm implementation
      scaling = scale

      Logger.info("[Bumblebee.Adapters] LoRA config: rank=#{rank}, scale=#{scale}")

      # Set EMLX backend
      Nx.default_backend({EMLX.Backend, device: device})

      # Find layer indices with adapters
      layer_indices = find_adapter_layers(header)
      Logger.info("[Bumblebee.Adapters] Found adapters for #{length(layer_indices)} layers")

      # Load adapters for each layer
      layers =
        for layer_idx <- layer_indices, into: %{} do
          adapters = load_layer_adapters(safetensors_path, header, layer_idx, device)
          {layer_idx, adapters}
        end

      adapter = %{
        layers: layers,
        config: config,
        scaling: scaling,
        rank: rank
      }

      {:ok, adapter}
    end
  end

  @doc """
  Applies LoRA to a base model output.

  This is the core operation for runtime LoRA application:

      output = base_output + scaling × (input @ lora_a @ lora_b)

  ## Parameters

    * `base_output` - Output from the base model's weight
    * `input` - Input to the layer (before base weight application)
    * `lora_pair` - The LoRA A/B matrices
    * `scaling` - Scaling factor from adapter config
  """
  @spec apply(Nx.Tensor.t(), Nx.Tensor.t(), lora_pair() | nil, number()) :: Nx.Tensor.t()
  def apply(base_output, _input, nil, _scaling) do
    # No adapter for this weight, return base output unchanged
    base_output
  end

  def apply(base_output, input, %{lora_a: lora_a, lora_b: lora_b}, scaling) do
    # LoRA: output = base_output + scaling × (input @ A @ B)
    lora_output =
      input
      |> Nx.dot(lora_a)
      |> Nx.dot(lora_b)
      |> Nx.multiply(scaling)

    Nx.add(base_output, lora_output)
  end

  @doc """
  Gets the adapter for a specific layer, or nil if none exists.
  """
  @spec get_layer_adapter(adapter(), non_neg_integer()) :: layer_adapters() | nil
  def get_layer_adapter(adapter, layer_idx) do
    Map.get(adapter.layers, layer_idx)
  end

  # ============================================================================
  # Training Adapters
  # ============================================================================

  @doc """
  Prepares training data in the format expected by mlx_lm.lora.

  Takes a list of content strings and writes them as ChatML JSONL.

  ## Example

      posts = ["First post content", "Second post content", ...]
      Bumblebee.Adapters.prepare_training_data(posts, "/path/to/data",
        prompt: "Write a post",
        split_ratio: 0.9
      )

  This creates:
  - `/path/to/data/train.jsonl`
  - `/path/to/data/valid.jsonl`
  """
  @spec prepare_training_data([String.t()], Path.t(), keyword()) :: :ok | {:error, term()}
  def prepare_training_data(contents, output_dir, opts \\ []) do
    prompt = Keyword.get(opts, :prompt, "Write a post")
    split_ratio = Keyword.get(opts, :split_ratio, 0.9)

    File.mkdir_p!(output_dir)

    # Convert to ChatML format
    messages =
      Enum.map(contents, fn content ->
        %{
          "messages" => [
            %{"role" => "user", "content" => prompt},
            %{"role" => "assistant", "content" => content}
          ]
        }
      end)

    # Shuffle and split
    shuffled = Enum.shuffle(messages)
    split_idx = round(length(shuffled) * split_ratio)
    {train, valid} = Enum.split(shuffled, split_idx)

    # Write JSONL files
    write_jsonl(Path.join(output_dir, "train.jsonl"), train)
    write_jsonl(Path.join(output_dir, "valid.jsonl"), valid)

    Logger.info("[Bumblebee.Adapters] Wrote #{length(train)} training, #{length(valid)} validation samples")
    :ok
  end

  @doc """
  Trains a LoRA adapter using mlx_lm.lora.

  This shells out to Python's mlx_lm package for training. Requires:
  - Python 3.10+
  - mlx_lm package (`pip install mlx_lm`)

  ## Options

    * `:base_model` - Path or HF repo of base model (required)
    * `:training_data` - Path to training data directory (required)
    * `:output_path` - Where to save the adapter (required)
    * `:iterations` - Training iterations (default: 1000)
    * `:rank` - LoRA rank (default: 8)
    * `:scale` - LoRA scale (default: 20.0)
    * `:learning_rate` - Learning rate (default: 1e-5)
    * `:batch_size` - Batch size (default: 1)
    * `:grad_accumulation` - Gradient accumulation steps (default: 8)

  ## Returns

  `{:ok, output_path}` on success, `{:error, reason}` on failure.
  """
  @spec train(keyword()) :: {:ok, Path.t()} | {:error, term()}
  def train(opts) do
    base_model = Keyword.fetch!(opts, :base_model)
    training_data = Keyword.fetch!(opts, :training_data)
    output_path = Keyword.fetch!(opts, :output_path)

    iterations = Keyword.get(opts, :iterations, 1000)
    rank = Keyword.get(opts, :rank, 8)
    scale = Keyword.get(opts, :scale, 20.0)
    learning_rate = Keyword.get(opts, :learning_rate, 1.0e-5)
    batch_size = Keyword.get(opts, :batch_size, 1)
    grad_accumulation = Keyword.get(opts, :grad_accumulation, 8)

    # Create config file
    config = %{
      "model" => base_model,
      "data" => training_data,
      "adapter_path" => output_path,
      "train" => true,
      "fine_tune_type" => "lora",
      "batch_size" => batch_size,
      "grad_accumulation_steps" => grad_accumulation,
      "iters" => iterations,
      "learning_rate" => learning_rate,
      "max_seq_length" => 256,
      "save_every" => div(iterations, 10),
      "steps_per_report" => 100,
      "steps_per_eval" => div(iterations, 10),
      "mask_prompt" => true,
      "seed" => 42,
      "lora_parameters" => %{
        "rank" => rank,
        "dropout" => 0.0,
        "scale" => scale
      }
    }

    config_path = Path.join(System.tmp_dir!(), "lora_config_#{:rand.uniform(100_000)}.yaml")
    File.write!(config_path, Jason.encode!(config))

    Logger.info("[Bumblebee.Adapters] Starting LoRA training...")
    Logger.info("[Bumblebee.Adapters] Config: #{config_path}")

    case System.cmd("python", ["-m", "mlx_lm.lora", "--config", config_path],
           stderr_to_stdout: true,
           into: IO.stream(:stdio, :line)
         ) do
      {_, 0} ->
        File.rm(config_path)
        Logger.info("[Bumblebee.Adapters] Training complete!")
        {:ok, output_path}

      {output, exit_code} ->
        File.rm(config_path)
        {:error, {:training_failed, exit_code, output}}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp load_config(adapter_dir) do
    config_path = Path.join(adapter_dir, "adapter_config.json")

    with {:ok, data} <- File.read(config_path),
         {:ok, config} <- Jason.decode(data) do
      {:ok, config}
    else
      {:error, :enoent} -> {:error, {:config_not_found, config_path}}
      {:error, reason} -> {:error, {:config_parse_error, reason}}
    end
  end

  defp find_adapter_layers(header) do
    header
    |> Map.keys()
    |> Enum.filter(&String.starts_with?(&1, "model.layers."))
    |> Enum.map(fn key ->
      case Regex.run(~r/model\.layers\.(\d+)\./, key) do
        [_, idx] -> String.to_integer(idx)
        _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
    |> Enum.sort()
  end

  defp load_layer_adapters(path, header, layer_idx, device) do
    prefix = "model.layers.#{layer_idx}"

    %{
      self_attn: %{
        q_proj: load_lora_pair(path, header, "#{prefix}.self_attn.q_proj", device),
        k_proj: load_lora_pair(path, header, "#{prefix}.self_attn.k_proj", device),
        v_proj: load_lora_pair(path, header, "#{prefix}.self_attn.v_proj", device),
        o_proj: load_lora_pair(path, header, "#{prefix}.self_attn.o_proj", device)
      },
      mlp: %{
        gate_proj: load_lora_pair(path, header, "#{prefix}.mlp.gate_proj", device),
        up_proj: load_lora_pair(path, header, "#{prefix}.mlp.up_proj", device),
        down_proj: load_lora_pair(path, header, "#{prefix}.mlp.down_proj", device)
      }
    }
  end

  defp load_lora_pair(path, header, base_name, device) do
    lora_a_name = "#{base_name}.lora_a"
    lora_b_name = "#{base_name}.lora_b"

    if Map.has_key?(header, lora_a_name) and Map.has_key?(header, lora_b_name) do
      %{
        lora_a: load_tensor(path, header, lora_a_name, device),
        lora_b: load_tensor(path, header, lora_b_name, device)
      }
    else
      nil
    end
  end

  defp load_tensor(path, header, name, device) do
    case Safetensors.load_tensor(path, header, name) do
      {:ok, tensor} ->
        Nx.backend_transfer(tensor, {EMLX.Backend, device: device})

      {:error, reason} ->
        Logger.warning("[Bumblebee.Adapters] Failed to load #{name}: #{inspect(reason)}")
        nil
    end
  end

  defp write_jsonl(path, items) do
    content =
      items
      |> Enum.map(&Jason.encode!/1)
      |> Enum.join("\n")

    File.write!(path, content <> "\n")
  end
end
