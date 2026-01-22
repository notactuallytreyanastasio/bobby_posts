defmodule BobbyPosts.QuantizedLoader do
  @moduledoc """
  Loads 4-bit quantized Qwen3 model weights from safetensors format.

  The MLX 4-bit format stores each weight matrix as three tensors:
  - `{name}.weight` - Packed uint32 (8 int4 values per uint32)
  - `{name}.scales` - BF16 per-group scale factors
  - `{name}.biases` - BF16 per-group zero points

  With group_size=64, for a weight matrix of shape [out, in]:
  - weight shape: [out, in/8]
  - scales shape: [out, in/64]
  - biases shape: [out, in/64]
  """

  require Logger

  alias BobbyPosts.Safetensors

  @type quantized_weight :: %{
          weight: Nx.Tensor.t(),
          scales: Nx.Tensor.t(),
          biases: Nx.Tensor.t()
        }

  @type layer_weights :: %{
          self_attn: %{
            q_proj: quantized_weight(),
            k_proj: quantized_weight(),
            v_proj: quantized_weight(),
            o_proj: quantized_weight()
          },
          mlp: %{
            gate_proj: quantized_weight(),
            up_proj: quantized_weight(),
            down_proj: quantized_weight()
          },
          input_layernorm: Nx.Tensor.t(),
          post_attention_layernorm: Nx.Tensor.t()
        }

  @doc """
  Loads the Qwen3 model configuration from config.json.
  """
  @spec load_config(Path.t()) :: {:ok, map()} | {:error, term()}
  def load_config(model_dir) do
    config_path = Path.join(model_dir, "config.json")

    with {:ok, data} <- File.read(config_path),
         {:ok, config} <- Jason.decode(data) do
      {:ok, config}
    end
  end

  @doc """
  Loads all model weights from safetensors file.

  Returns a map with:
  - :embed_tokens - Quantized embedding weights
  - :layers - List of layer_weights for each transformer layer
  - :norm - Final layer norm weight
  - :lm_head - Quantized output projection (may be tied to embed_tokens)
  - :config - Model configuration
  """
  @spec load_model(Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def load_model(model_dir, opts \\ []) do
    safetensors_path = Path.join(model_dir, "model.safetensors")
    device = Keyword.get(opts, :device, :gpu)

    Logger.info("Loading model from #{model_dir}")
    Logger.info("Device: #{device}")

    with {:ok, config} <- load_config(model_dir),
         {:ok, {header, _data_offset}} <- Safetensors.read_header(safetensors_path) do
      num_layers = config["num_hidden_layers"]
      Logger.info("Model has #{num_layers} layers")

      # Set EMLX as backend for GPU loading
      Nx.default_backend({EMLX.Backend, device: device})

      # Load embedding weights
      Logger.info("Loading embedding weights...")
      embed_tokens = load_quantized_weight(safetensors_path, header, "model.embed_tokens")

      # Load each transformer layer
      Logger.info("Loading transformer layers...")

      layers =
        for i <- 0..(num_layers - 1) do
          Logger.debug("Loading layer #{i}...")
          load_layer(safetensors_path, header, i)
        end

      # Load final norm
      Logger.info("Loading final norm...")
      norm = load_tensor_to_device(safetensors_path, header, "model.norm.weight", device)

      # Load lm_head (output projection)
      Logger.info("Loading lm_head...")
      lm_head = load_quantized_weight(safetensors_path, header, "lm_head")

      model = %{
        embed_tokens: embed_tokens,
        layers: layers,
        norm: norm,
        lm_head: lm_head,
        config: config
      }

      Logger.info("Model loaded successfully!")
      {:ok, model}
    end
  end

  @doc """
  Loads a single quantized weight (weight + scales + biases).
  """
  @spec load_quantized_weight(Path.t(), map(), String.t()) :: quantized_weight()
  def load_quantized_weight(path, header, base_name) do
    # Load each component
    weight = load_tensor_gpu(path, header, "#{base_name}.weight")
    scales = load_tensor_gpu(path, header, "#{base_name}.scales")
    biases = load_tensor_gpu(path, header, "#{base_name}.biases")

    %{weight: weight, scales: scales, biases: biases}
  end

  @doc """
  Loads a transformer layer's weights.
  """
  @spec load_layer(Path.t(), map(), non_neg_integer()) :: layer_weights()
  def load_layer(path, header, layer_idx) do
    prefix = "model.layers.#{layer_idx}"

    %{
      self_attn: %{
        q_proj: load_quantized_weight(path, header, "#{prefix}.self_attn.q_proj"),
        k_proj: load_quantized_weight(path, header, "#{prefix}.self_attn.k_proj"),
        v_proj: load_quantized_weight(path, header, "#{prefix}.self_attn.v_proj"),
        o_proj: load_quantized_weight(path, header, "#{prefix}.self_attn.o_proj"),
        # Qwen3 has Q/K normalization before RoPE
        q_norm: load_tensor_gpu(path, header, "#{prefix}.self_attn.q_norm.weight"),
        k_norm: load_tensor_gpu(path, header, "#{prefix}.self_attn.k_norm.weight")
      },
      mlp: %{
        gate_proj: load_quantized_weight(path, header, "#{prefix}.mlp.gate_proj"),
        up_proj: load_quantized_weight(path, header, "#{prefix}.mlp.up_proj"),
        down_proj: load_quantized_weight(path, header, "#{prefix}.mlp.down_proj")
      },
      input_layernorm: load_tensor_gpu(path, header, "#{prefix}.input_layernorm.weight"),
      post_attention_layernorm:
        load_tensor_gpu(path, header, "#{prefix}.post_attention_layernorm.weight")
    }
  end

  # Load a tensor and transfer to GPU
  defp load_tensor_gpu(path, header, name) do
    case Safetensors.load_tensor(path, header, name) do
      {:ok, tensor} ->
        # Transfer to GPU backend
        Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})

      {:error, reason} ->
        Logger.warning("Failed to load tensor #{name}: #{inspect(reason)}")
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

  @doc """
  Gets memory statistics for loaded model.
  """
  @spec memory_stats(map()) :: map()
  def memory_stats(model) do
    layer_count = length(model.layers)

    # Calculate approximate memory usage
    # Each layer has attention (4 weights) + MLP (3 weights) + 2 norms
    layer_params =
      model.layers
      |> Enum.take(1)
      |> Enum.map(fn layer ->
        count_quantized_params(layer.self_attn.q_proj) +
          count_quantized_params(layer.self_attn.k_proj) +
          count_quantized_params(layer.self_attn.v_proj) +
          count_quantized_params(layer.self_attn.o_proj) +
          count_quantized_params(layer.mlp.gate_proj) +
          count_quantized_params(layer.mlp.up_proj) +
          count_quantized_params(layer.mlp.down_proj) +
          tensor_size(layer.input_layernorm) +
          tensor_size(layer.post_attention_layernorm)
      end)
      |> Enum.sum()

    embed_params = count_quantized_params(model.embed_tokens)
    lm_head_params = count_quantized_params(model.lm_head)
    norm_params = tensor_size(model.norm)

    total_params = embed_params + lm_head_params + norm_params + layer_params * layer_count

    %{
      layer_count: layer_count,
      params_per_layer: layer_params,
      embed_params: embed_params,
      lm_head_params: lm_head_params,
      total_params: total_params,
      # Rough memory estimate (most weights are 4-bit + some BF16)
      estimated_memory_gb: total_params / (1024 * 1024 * 1024)
    }
  end

  defp count_quantized_params(%{weight: w, scales: s, biases: b}) do
    tensor_size(w) + tensor_size(s) + tensor_size(b)
  end

  defp count_quantized_params(nil), do: 0

  defp tensor_size(nil), do: 0

  defp tensor_size(tensor) do
    {_, bits} = Nx.type(tensor)
    Nx.size(tensor) * div(bits, 8)
  end
end
