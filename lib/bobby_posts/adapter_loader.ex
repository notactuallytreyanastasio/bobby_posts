defmodule BobbyPosts.AdapterLoader do
  @moduledoc """
  Loads LoRA adapter weights from safetensors format.

  LoRA adapters consist of two small matrices per weight:
  - lora_a: [input_dim, rank] - down projection
  - lora_b: [rank, output_dim] - up projection

  During forward pass: output = base_output + (x @ lora_a @ lora_b) * scaling
  """

  require Logger

  alias BobbyPosts.Safetensors

  @doc """
  Loads adapter config from adapter_config.json.
  """
  @spec load_config(Path.t()) :: {:ok, map()} | {:error, term()}
  def load_config(adapter_dir) do
    config_path = Path.join(adapter_dir, "adapter_config.json")

    with {:ok, data} <- File.read(config_path),
         {:ok, config} <- Jason.decode(data) do
      {:ok, config}
    end
  end

  @doc """
  Loads all LoRA adapter weights from safetensors.

  Returns a map with:
  - :layers - Map of layer_idx => layer adapters
  - :config - Adapter configuration (rank, scale, etc.)
  - :scaling - Pre-computed scaling factor (scale / rank)
  """
  @spec load_adapters(Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def load_adapters(adapter_dir, opts \\ []) do
    safetensors_path = Path.join(adapter_dir, "adapters.safetensors")
    device = Keyword.get(opts, :device, :gpu)

    Logger.info("Loading adapters from #{adapter_dir}")

    with {:ok, config} <- load_config(adapter_dir),
         {:ok, {header, _data_offset}} <- Safetensors.read_header(safetensors_path) do

      lora_params = config["lora_parameters"]
      rank = lora_params["rank"]
      scale = lora_params["scale"]
      # NOTE: mlx_lm uses scale directly, NOT scale/rank
      # The standard LoRA paper uses alpha/rank, but mlx_lm doesn't divide by rank
      scaling = scale

      Logger.info("LoRA config: rank=#{rank}, scale=#{scale}, scaling=#{scaling}")

      # Set EMLX as backend for GPU loading
      Nx.default_backend({EMLX.Backend, device: device})

      # Find all layer indices that have adapters
      layer_indices =
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

      Logger.info("Found adapters for #{length(layer_indices)} layers: #{inspect(layer_indices)}")

      # Load adapters for each layer
      layers =
        for layer_idx <- layer_indices, into: %{} do
          adapters = load_layer_adapters(safetensors_path, header, layer_idx, device)
          {layer_idx, adapters}
        end

      result = %{
        layers: layers,
        config: config,
        scaling: scaling,
        rank: rank
      }

      Logger.info("Adapters loaded successfully!")
      {:ok, result}
    end
  end

  @doc """
  Loads LoRA adapters for a single layer.
  """
  def load_layer_adapters(path, header, layer_idx, device) do
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

  @doc """
  Loads a LoRA A/B pair for a single weight.
  """
  def load_lora_pair(path, header, base_name, device) do
    lora_a_name = "#{base_name}.lora_a"
    lora_b_name = "#{base_name}.lora_b"

    # Check if both exist
    if Map.has_key?(header, lora_a_name) and Map.has_key?(header, lora_b_name) do
      lora_a = load_tensor_to_device(path, header, lora_a_name, device)
      lora_b = load_tensor_to_device(path, header, lora_b_name, device)
      %{lora_a: lora_a, lora_b: lora_b}
    else
      nil
    end
  end

  defp load_tensor_to_device(path, header, name, device) do
    case Safetensors.load_tensor(path, header, name) do
      {:ok, tensor} ->
        Nx.backend_transfer(tensor, {EMLX.Backend, device: device})

      {:error, reason} ->
        Logger.warning("Failed to load tensor #{name}: #{inspect(reason)}")
        nil
    end
  end
end
