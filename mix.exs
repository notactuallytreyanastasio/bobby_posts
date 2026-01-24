defmodule BobbyPosts.MixProject do
  use Mix.Project

  def project do
    [
      app: :bobby_posts,
      version: "0.1.0",
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps(),
      compilers: [:phoenix_live_view] ++ Mix.compilers(),
      listeners: [Phoenix.CodeReloader]
    ]
  end

  # Configuration for the OTP application.
  #
  # Type `mix help compile.app` for more information.
  def application do
    [
      mod: {BobbyPosts.Application, []},
      extra_applications: [:logger, :runtime_tools]
    ]
  end

  def cli do
    [
      preferred_envs: [precommit: :test]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Specifies your project dependencies.
  #
  # Type `mix help deps` for examples and options.
  defp deps do
    [
      # Phoenix Web
      {:phoenix, "~> 1.8.3"},
      {:phoenix_html, "~> 4.1"},
      {:phoenix_live_reload, "~> 1.2", only: :dev},
      {:phoenix_live_view, "~> 1.1.0"},
      {:lazy_html, ">= 0.1.0", only: :test},
      {:esbuild, "~> 0.10", runtime: Mix.env() == :dev},
      {:tailwind, "~> 0.3", runtime: Mix.env() == :dev},
      {:heroicons,
       github: "tailwindlabs/heroicons",
       tag: "v2.2.0",
       sparse: "optimized",
       app: false,
       compile: false,
       depth: 1},
      {:telemetry_metrics, "~> 1.0"},
      {:telemetry_poller, "~> 1.0"},
      {:gettext, "~> 1.0"},
      {:jason, "~> 1.2"},
      {:dns_cluster, "~> 0.2.0"},
      {:bandit, "~> 1.5"},

      # ML Stack
      # TODO: After upstream PRs merge, update to:
      #   {:emlx, "~> 0.3"},      # After quantization PR
      #   {:safetensors, "~> 0.1"}, # After hex.pm publish
      #   {:bumblebee, "~> 0.6"},   # Qwen3 already in upstream
      {:nx, "~> 0.10"},
      {:axon, "~> 0.7"},
      {:emlx, github: "notactuallytreyanastasio/emlx", branch: "feat/quantization-ops"},
      {:safetensors, github: "notactuallytreyanastasio/safetensors_ex", override: true},
      # Note: Qwen3 is in upstream Bumblebee, using fork for consistency until all PRs merge
      {:bumblebee, github: "notactuallytreyanastasio/bumblebee", branch: "feat/qwen3"},

      # Utils
      {:dotenvy, "~> 0.8.0"}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project.
  # For example, to install project dependencies and perform other setup tasks, run:
  #
  #     $ mix setup
  #
  # See the documentation for `Mix` for more info on aliases.
  defp aliases do
    [
      setup: ["deps.get", "assets.setup", "assets.build"],
      "assets.setup": ["tailwind.install --if-missing", "esbuild.install --if-missing"],
      "assets.build": ["compile", "tailwind bobby_posts", "esbuild bobby_posts"],
      "assets.deploy": [
        "tailwind bobby_posts --minify",
        "esbuild bobby_posts --minify",
        "phx.digest"
      ],
      precommit: ["compile --warnings-as-errors", "deps.unlock --unused", "format", "test"]
    ]
  end
end
