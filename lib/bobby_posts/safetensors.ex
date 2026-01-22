defmodule BobbyPosts.Safetensors do
  @moduledoc """
  Safetensors file parser for MLX 4-bit quantized models.

  Safetensors format:
  - First 8 bytes: header size as u64 little-endian
  - Next N bytes: JSON header with tensor metadata
  - Remaining bytes: tensor data

  Each tensor entry in the header contains:
  - `dtype`: Data type (U32, BF16, F32, etc.)
  - `shape`: Tensor shape as list
  - `data_offsets`: [start, end] byte offsets in the data section
  """

  require Logger

  @type tensor_info :: %{
          dtype: String.t(),
          shape: [non_neg_integer()],
          data_offsets: [non_neg_integer()]
        }

  @type header :: %{String.t() => tensor_info()}

  @doc """
  Reads the header from a safetensors file.

  Returns {:ok, {header, data_offset}} where:
  - header: Map of tensor_name => tensor_info
  - data_offset: Byte offset where tensor data begins
  """
  @spec read_header(Path.t()) :: {:ok, {header(), non_neg_integer()}} | {:error, term()}
  def read_header(path) do
    with {:ok, file} <- File.open(path, [:read, :binary]),
         {:ok, header_len} <- read_header_length(file),
         {:ok, header} <- read_header_json(file, header_len) do
      File.close(file)
      {:ok, {header, 8 + header_len}}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Reads a specific tensor from a safetensors file.

  Returns the raw binary data for the tensor.
  """
  @spec read_tensor(Path.t(), header(), String.t()) :: {:ok, binary()} | {:error, term()}
  def read_tensor(path, header, tensor_name) do
    case Map.get(header, tensor_name) do
      nil ->
        {:error, {:tensor_not_found, tensor_name}}

      %{"data_offsets" => [start_offset, end_offset]} ->
        # Data starts after header (8 bytes header_len + header_len bytes JSON)
        {:ok, data_offset} = get_data_offset(path)

        with {:ok, file} <- File.open(path, [:read, :binary]),
             :ok <- position_file(file, data_offset + start_offset),
             {:ok, data} <- read_bytes(file, end_offset - start_offset) do
          File.close(file)
          {:ok, data}
        else
          {:error, reason} -> {:error, reason}
        end
    end
  end

  @doc """
  Reads multiple tensors from a safetensors file.

  More efficient than calling read_tensor multiple times as it reads
  tensors in order of their position in the file.
  """
  @spec read_tensors(Path.t(), header(), [String.t()]) ::
          {:ok, %{String.t() => binary()}} | {:error, term()}
  def read_tensors(path, header, tensor_names) do
    # Get tensor infos and sort by start offset for efficient reading
    tensors_with_info =
      tensor_names
      |> Enum.map(fn name ->
        case Map.get(header, name) do
          nil -> {name, nil}
          info -> {name, info}
        end
      end)
      |> Enum.filter(fn {_name, info} -> info != nil end)
      |> Enum.sort_by(fn {_name, %{"data_offsets" => [start, _]}} -> start end)

    missing = tensor_names -- Enum.map(tensors_with_info, fn {name, _} -> name end)

    if missing != [] do
      Logger.warning("Missing tensors: #{inspect(missing)}")
    end

    {:ok, data_offset} = get_data_offset(path)

    with {:ok, file} <- File.open(path, [:read, :binary]) do
      result =
        tensors_with_info
        |> Enum.reduce_while({:ok, %{}}, fn {name, %{"data_offsets" => [start_offset, end_offset]}},
                                             {:ok, acc} ->
          case read_tensor_at(file, data_offset + start_offset, end_offset - start_offset) do
            {:ok, data} -> {:cont, {:ok, Map.put(acc, name, data)}}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)

      File.close(file)
      result
    end
  end

  @doc """
  Gets the data type as an Nx type.
  """
  @spec dtype_to_nx(String.t()) :: {:ok, Nx.Type.t()} | {:error, term()}
  def dtype_to_nx("F32"), do: {:ok, {:f, 32}}
  def dtype_to_nx("F16"), do: {:ok, {:f, 16}}
  def dtype_to_nx("BF16"), do: {:ok, {:bf, 16}}
  def dtype_to_nx("I32"), do: {:ok, {:s, 32}}
  def dtype_to_nx("I64"), do: {:ok, {:s, 64}}
  def dtype_to_nx("U32"), do: {:ok, {:u, 32}}
  def dtype_to_nx("U64"), do: {:ok, {:u, 64}}
  def dtype_to_nx("I8"), do: {:ok, {:s, 8}}
  def dtype_to_nx("U8"), do: {:ok, {:u, 8}}
  def dtype_to_nx(dtype), do: {:error, {:unsupported_dtype, dtype}}

  @doc """
  Converts binary data to an Nx tensor.
  """
  @spec to_nx_tensor(binary(), String.t(), [non_neg_integer()]) ::
          {:ok, Nx.Tensor.t()} | {:error, term()}
  def to_nx_tensor(data, dtype, shape) do
    with {:ok, nx_type} <- dtype_to_nx(dtype) do
      tensor =
        data
        |> Nx.from_binary(nx_type)
        |> Nx.reshape(List.to_tuple(shape))

      {:ok, tensor}
    end
  end

  @doc """
  Reads a tensor and converts it to Nx tensor.
  """
  @spec load_tensor(Path.t(), header(), String.t()) ::
          {:ok, Nx.Tensor.t()} | {:error, term()}
  def load_tensor(path, header, tensor_name) do
    case Map.get(header, tensor_name) do
      nil ->
        {:error, {:tensor_not_found, tensor_name}}

      %{"data_offsets" => _, "dtype" => dtype, "shape" => shape} = _info ->
        with {:ok, data} <- read_tensor(path, header, tensor_name),
             {:ok, tensor} <- to_nx_tensor(data, dtype, shape) do
          {:ok, tensor}
        end
    end
  end

  # Private functions

  defp read_header_length(file) do
    case IO.binread(file, 8) do
      <<header_len::little-unsigned-64>> -> {:ok, header_len}
      :eof -> {:error, :unexpected_eof}
      {:error, reason} -> {:error, reason}
    end
  end

  defp read_header_json(file, header_len) do
    case IO.binread(file, header_len) do
      data when is_binary(data) and byte_size(data) == header_len ->
        case Jason.decode(data) do
          {:ok, header} ->
            # Remove metadata key
            {:ok, Map.delete(header, "__metadata__")}

          {:error, reason} ->
            {:error, {:json_decode_error, reason}}
        end

      :eof ->
        {:error, :unexpected_eof}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp get_data_offset(path) do
    with {:ok, file} <- File.open(path, [:read, :binary]),
         {:ok, header_len} <- read_header_length(file) do
      File.close(file)
      {:ok, 8 + header_len}
    end
  end

  defp position_file(file, position) do
    case :file.position(file, position) do
      {:ok, ^position} -> :ok
      {:ok, _other} -> {:error, :position_failed}
      {:error, reason} -> {:error, reason}
    end
  end

  defp read_bytes(file, count) do
    case IO.binread(file, count) do
      data when is_binary(data) and byte_size(data) == count -> {:ok, data}
      data when is_binary(data) -> {:error, {:short_read, byte_size(data), count}}
      :eof -> {:error, :unexpected_eof}
      {:error, reason} -> {:error, reason}
    end
  end

  defp read_tensor_at(file, position, count) do
    with :ok <- position_file(file, position) do
      read_bytes(file, count)
    end
  end
end
