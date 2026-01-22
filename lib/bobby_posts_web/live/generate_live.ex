defmodule BobbyPostsWeb.GenerateLive do
  use BobbyPostsWeb, :live_view

  require Logger

  @impl true
  def mount(_params, _session, socket) do
    # Start the generator if not already running
    case Process.whereis(BobbyPosts.Generator) do
      nil -> BobbyPosts.Generator.start_link()
      _pid -> :ok
    end

    {:ok, assign(socket,
      prompt: "",
      max_tokens: 150,
      count: 1,
      posts: [],
      generating: false,
      error: nil,
      model_loaded: false
    )}
  end

  @impl true
  def handle_event("update_form", %{"prompt" => prompt, "max_tokens" => max_tokens, "count" => count}, socket) do
    {:noreply, assign(socket,
      prompt: prompt,
      max_tokens: parse_int(max_tokens, 150),
      count: parse_int(count, 1)
    )}
  end

  @impl true
  def handle_event("generate", _params, socket) do
    # Start async generation
    socket = assign(socket, generating: true, error: nil, posts: [])

    # Run generation in a task to not block the LiveView
    prompt = if socket.assigns.prompt == "", do: nil, else: socket.assigns.prompt
    max_tokens = socket.assigns.max_tokens
    count = socket.assigns.count

    pid = self()

    Task.start(fn ->
      result = BobbyPosts.Generator.generate(
        prompt: prompt,
        max_tokens: max_tokens,
        count: count
      )
      send(pid, {:generation_complete, result})
    end)

    {:noreply, socket}
  end

  @impl true
  def handle_event("copy_post", %{"index" => index_str}, socket) do
    index = String.to_integer(index_str)
    post = Enum.at(socket.assigns.posts, index)

    # Send JS command to copy to clipboard
    {:noreply, push_event(socket, "copy_to_clipboard", %{text: post})}
  end

  @impl true
  def handle_info({:generation_complete, {:ok, posts}}, socket) do
    {:noreply, assign(socket, generating: false, posts: posts, model_loaded: true)}
  end

  @impl true
  def handle_info({:generation_complete, {:error, reason}}, socket) do
    {:noreply, assign(socket, generating: false, error: inspect(reason))}
  end

  defp parse_int(str, default) do
    case Integer.parse(str) do
      {num, _} -> num
      :error -> default
    end
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-base-200 py-8">
      <div class="container mx-auto max-w-4xl px-4">
        <div class="text-center mb-8">
          <h1 class="text-4xl font-bold mb-2">Bobby Posts Generator</h1>
          <p class="text-base-content/70">Generate posts in Bobby's voice using Qwen3-8B</p>
        </div>

        <div class="card bg-base-100 shadow-xl mb-8">
          <div class="card-body">
            <form phx-change="update_form" phx-submit="generate">
              <div class="form-control mb-4">
                <label class="label">
                  <span class="label-text font-semibold">Prompt (optional)</span>
                </label>
                <textarea
                  name="prompt"
                  class="textarea textarea-bordered h-24"
                  placeholder="What's your take on functional programming?"
                  value={@prompt}
                ><%= @prompt %></textarea>
                <label class="label">
                  <span class="label-text-alt">Leave empty for open-ended generation</span>
                </label>
              </div>

              <div class="grid grid-cols-2 gap-4 mb-6">
                <div class="form-control">
                  <label class="label">
                    <span class="label-text font-semibold">Max Tokens</span>
                  </label>
                  <input
                    type="number"
                    name="max_tokens"
                    class="input input-bordered"
                    value={@max_tokens}
                    min="10"
                    max="500"
                  />
                </div>

                <div class="form-control">
                  <label class="label">
                    <span class="label-text font-semibold">Count</span>
                  </label>
                  <input
                    type="number"
                    name="count"
                    class="input input-bordered"
                    value={@count}
                    min="1"
                    max="10"
                  />
                </div>
              </div>

              <button
                type="submit"
                class={"btn btn-primary btn-lg w-full " <> if(@generating, do: "loading", else: "")}
                disabled={@generating}
              >
                <%= if @generating do %>
                  Generating...
                <% else %>
                  Generate Post
                <% end %>
              </button>
            </form>
          </div>
        </div>

        <%= if @error do %>
          <div class="alert alert-error mb-8">
            <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Error: <%= @error %></span>
          </div>
        <% end %>

        <%= if @posts != [] do %>
          <div class="space-y-4">
            <h2 class="text-2xl font-bold">Generated Posts</h2>

            <%= for {post, index} <- Enum.with_index(@posts) do %>
              <div class="card bg-base-100 shadow-lg">
                <div class="card-body">
                  <p class="whitespace-pre-wrap text-lg"><%= post %></p>
                  <div class="card-actions justify-end mt-4">
                    <button
                      class="btn btn-sm btn-outline"
                      phx-click="copy_post"
                      phx-value-index={index}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Copy
                    </button>
                  </div>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>

        <%= if @generating do %>
          <div class="text-center py-8">
            <span class="loading loading-spinner loading-lg"></span>
            <p class="mt-4 text-base-content/70">
              <%= if @model_loaded do %>
                Generating posts...
              <% else %>
                Loading model (this may take a moment on first run)...
              <% end %>
            </p>
          </div>
        <% end %>
      </div>
    </div>

    <script>
      window.addEventListener("phx:copy_to_clipboard", (event) => {
        navigator.clipboard.writeText(event.detail.text).then(() => {
          // Could show a toast notification here
          console.log("Copied to clipboard");
        });
      });
    </script>
    """
  end
end
