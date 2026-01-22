defmodule BobbyPosts.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      BobbyPostsWeb.Telemetry,
      {DNSCluster, query: Application.get_env(:bobby_posts, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: BobbyPosts.PubSub},
      # Generator holds the loaded Qwen3 model
      BobbyPosts.Generator,
      # Start to serve requests, typically the last entry
      BobbyPostsWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: BobbyPosts.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    BobbyPostsWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
