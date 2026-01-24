# This file is responsible for configuring your application
# and its dependencies with the aid of the Config module.
#
# This configuration file is loaded before any dependency and
# is restricted to this project.

# General application configuration
import Config

config :bobby_posts,
  generators: [timestamp_type: :utc_datetime],
  # Base 4-bit quantized model (NOT fused - apply LoRA at runtime for better quality)
  model_path: "/Users/robertgrayson/.cache/huggingface/hub/models--lmstudio-community--Qwen3-8B-MLX-4bit/snapshots/a84107f5c4dfdecf389b208598faeac322048237",
  # LoRA adapters applied at runtime in fp32 for full fine-tuning precision
  adapter_path: "/Users/robertgrayson/twitter_finetune/adapters_qwen3_4bit_v3"

# Configure the endpoint
config :bobby_posts, BobbyPostsWeb.Endpoint,
  url: [host: "localhost"],
  adapter: Bandit.PhoenixAdapter,
  render_errors: [
    formats: [html: BobbyPostsWeb.ErrorHTML, json: BobbyPostsWeb.ErrorJSON],
    layout: false
  ],
  pubsub_server: BobbyPosts.PubSub,
  live_view: [signing_salt: "2+V8sQNY"]

# Configure esbuild (the version is required)
config :esbuild,
  version: "0.25.4",
  bobby_posts: [
    args:
      ~w(js/app.js --bundle --target=es2022 --outdir=../priv/static/assets/js --external:/fonts/* --external:/images/* --alias:@=.),
    cd: Path.expand("../assets", __DIR__),
    env: %{"NODE_PATH" => [Path.expand("../deps", __DIR__), Mix.Project.build_path()]}
  ]

# Configure tailwind (the version is required)
config :tailwind,
  version: "4.1.12",
  bobby_posts: [
    args: ~w(
      --input=assets/css/app.css
      --output=priv/static/assets/css/app.css
    ),
    cd: Path.expand("..", __DIR__)
  ]

# Configure Elixir's Logger
config :logger, :default_formatter,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id]

# Use Jason for JSON parsing in Phoenix
config :phoenix, :json_library, Jason

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{config_env()}.exs"
