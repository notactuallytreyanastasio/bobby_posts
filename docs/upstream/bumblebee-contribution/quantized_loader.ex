defmodule Bumblebee.QuantizedLoader do
  @moduledoc """
  Loads 4-bit quantized model weights from MLX safetensors format.

  This loader is designed for Apple Silicon only, using the EMLX backend
  with MLX's optimized quantized_matmul kernels.

  ## Quantized Format

  MLX 4-bit format stores each weight matrix as three tensors:
  - `{name}.weight` - Packed uint32 (8 int4 values per uint32)
  - `{name}.scales` - BF16 per-group scale factors
  - `{name}.biases` - BF16 per-group zero points

  With group_size=64, for a weight matrix of shape [out, in]:
  - weight shape: [out, in/8]
  - scales shape: [out, in/64]
  - biases shape: [out, in/64]

  ## Example

      {:ok, model} = Bumblebee.QuantizedLoader.load_model(
        "/path/to/Qwen3-8B-MLX-4bit",
        architecture: :qwen3,
        device: :gpu
      )

  ## Requirements

  - Apple Silicon Mac (M1/M2/M3/M4)
  - EMLX backend with quantization ops
  - Model in MLX safetensors format (e.g., from lmstudio-community)
  """

  require Logger

  @type quantized_weight :: %{
          weight: Nx.Tensor.t(),
          scales: Nx.Tensor.t(),
          biases: Nx.Tensor.t()
        }

  @type model_spec :: %{
          embed_tokens: quantized_weight(),
          layers: [layer_weights()],
          norm: Nx.Tensor.t(),
          lm_head: quantized_weight(),
          config: map()
        }

  @type layer_weights :: %{
          self_attn: attention_weights(),
          mlp: mlp_weights(),
          input_layernorm: Nx.Tensor.t(),
          post_attention_layernorm: Nx.Tensor.t()
        }

  @type attention_weights :: %{
          q_proj: quantized_weight(),
          k_proj: quantized_weight(),
          v_proj: quantized_weight(),
          o_proj: quantized_weight(),
          q_norm: Nx.Tensor.t() | nil,
          k_norm: Nx.Tensor.t() | nil
        }

  @type mlp_weights :: %{
          gate_proj: quantized_weight(),
          up_proj: quantized_weight(),
          down_proj: quantized_weight()
        }

  @doc """
  Loads a quantized model from a local directory.

  ## Options

    * `:architecture` - Model architecture (`:qwen3`, `:llama`, etc.)
    * `:device` - Device to load to (`:gpu` or `:cpu`, default: `:gpu`)
    * `:log_level` - Logging level (`:debug`, `:info`, `:warning`, default: `:info`)

  ## Returns

  `{:ok, model_spec}` on success, `{:error, reason}` on failure.
  """
  @spec load_model(Path.t(), keyword()) :: {:ok, model_spec()} | {:error, term()}
  def load_model(model_dir, opts \\ []) do
    device = Keyword.get(opts, :device, :gpu)
    architecture = Keyword.get(opts, :architecture, :auto)

    with :ok <- validate_backend(),
         {:ok, config} <- load_config(model_dir),
         {:ok, arch} <- resolve_architecture(architecture, config),
         {:ok, header, path} <- load_safetensors_header(model_dir) do

      Logger.info("[Bumblebee.QuantizedLoader] Loading #{arch} model from #{model_dir}")
      Logger.info("[Bumblebee.QuantizedLoader] Device: #{device}")

      # Set EMLX backend
      Nx.default_backend({EMLX.Backend, device: device})

      model = load_model_weights(arch, path, header, config, device)

      Logger.info("[Bumblebee.QuantizedLoader] Model loaded successfully!")
      {:ok, model}
    end
  end

  @doc """
  Validates that EMLX backend is available with quantization support.
  """
  @spec validate_backend() :: :ok | {:error, term()}
  def validate_backend do
    cond do
      not Code.ensure_loaded?(EMLX) ->
        {:error, :emlx_not_available}

      not function_exported?(EMLX, :quantized_matmul, 7) ->
        {:error, :quantized_matmul_not_available}

      true ->
        :ok
    end
  end

  @doc """
  Loads a quantized weight triplet (weight + scales + biases).
  """
  @spec load_quantized_weight(Path.t(), map(), String.t(), atom()) :: quantized_weight()
  def load_quantized_weight(path, header, base_name, device) do
    %{
      weight: load_tensor(path, header, "#{base_name}.weight", device),
      scales: load_tensor(path, header, "#{base_name}.scales", device),
      biases: load_tensor(path, header, "#{base_name}.biases", device)
    }
  end

  # Private functions

  defp load_config(model_dir) do
    config_path = Path.join(model_dir, "config.json")

    with {:ok, data} <- File.read(config_path),
         {:ok, config} <- Jason.decode(data) do
      {:ok, config}
    else
      {:error, :enoent} -> {:error, {:config_not_found, config_path}}
      {:error, reason} -> {:error, {:config_parse_error, reason}}
    end
  end

  defp resolve_architecture(:auto, config) do
    case config["model_type"] do
      "qwen3" -> {:ok, :qwen3}
      "qwen2" -> {:ok, :qwen3}  # Compatible architecture
      "llama" -> {:ok, :llama}
      "mistral" -> {:ok, :mistral}
      other -> {:error, {:unknown_architecture, other}}
    end
  end

  defp resolve_architecture(arch, _config) when is_atom(arch), do: {:ok, arch}

  defp load_safetensors_header(model_dir) do
    path = Path.join(model_dir, "model.safetensors")

    case Safetensors.read_header(path) do
      {:ok, {header, _data_offset}} -> {:ok, header, path}
      {:error, reason} -> {:error, {:safetensors_error, reason}}
    end
  end

  defp load_model_weights(:qwen3, path, header, config, device) do
    num_layers = config["num_hidden_layers"]
    Logger.info("[Bumblebee.QuantizedLoader] Loading #{num_layers} transformer layers...")

    %{
      embed_tokens: load_quantized_weight(path, header, "model.embed_tokens", device),
      layers: load_transformer_layers(:qwen3, path, header, num_layers, device),
      norm: load_tensor(path, header, "model.norm.weight", device),
      lm_head: load_quantized_weight(path, header, "lm_head", device),
      config: config,
      architecture: :qwen3
    }
  end

  defp load_model_weights(arch, _path, _header, _config, _device) do
    raise "Architecture #{arch} not yet supported for quantized loading"
  end

  defp load_transformer_layers(:qwen3, path, header, num_layers, device) do
    for i <- 0..(num_layers - 1) do
      prefix = "model.layers.#{i}"

      %{
        self_attn: %{
          q_proj: load_quantized_weight(path, header, "#{prefix}.self_attn.q_proj", device),
          k_proj: load_quantized_weight(path, header, "#{prefix}.self_attn.k_proj", device),
          v_proj: load_quantized_weight(path, header, "#{prefix}.self_attn.v_proj", device),
          o_proj: load_quantized_weight(path, header, "#{prefix}.self_attn.o_proj", device),
          # Qwen3 has Q/K normalization before RoPE
          q_norm: load_tensor(path, header, "#{prefix}.self_attn.q_norm.weight", device),
          k_norm: load_tensor(path, header, "#{prefix}.self_attn.k_norm.weight", device)
        },
        mlp: %{
          gate_proj: load_quantized_weight(path, header, "#{prefix}.mlp.gate_proj", device),
          up_proj: load_quantized_weight(path, header, "#{prefix}.mlp.up_proj", device),
          down_proj: load_quantized_weight(path, header, "#{prefix}.mlp.down_proj", device)
        },
        input_layernorm: load_tensor(path, header, "#{prefix}.input_layernorm.weight", device),
        post_attention_layernorm: load_tensor(path, header, "#{prefix}.post_attention_layernorm.weight", device)
      }
    end
  end

  defp load_tensor(path, header, name, device) do
    case Safetensors.load_tensor(path, header, name) do
      {:ok, tensor} ->
        Nx.backend_transfer(tensor, {EMLX.Backend, device: device})

      {:error, reason} ->
        Logger.warning("[Bumblebee.QuantizedLoader] Failed to load tensor #{name}: #{inspect(reason)}")
        nil
    end
  end
end
