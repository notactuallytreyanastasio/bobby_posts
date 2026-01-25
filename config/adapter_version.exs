# Adapter version tracking
# This file records which adapter version the code is configured to use
# Update this when switching adapters and commit it with the change

import Config

config :bobby_posts,
  adapter_version: "v5",
  adapter_path: "/Users/robertgrayson/twitter_finetune/adapters_qwen3_4bit_v5",
  training_config: "qwen3_4bit_v5_config.yaml",
  training_params: %{
    iterations: 25_000,
    learning_rate: 1.0e-5,
    lora_rank: 8,
    lora_scale: 20.0,
    min_post_length: 160,
    bluesky_weight: 3,
    avg_training_length: 230
  }
