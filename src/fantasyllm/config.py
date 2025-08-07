"""Configuration management for Fantasy LLM."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Model Configuration
    base_model: str = Field(default="meta-llama/Llama-2-7b-chat-hf", env="BASE_MODEL")
    adapter_dir: Path = Field(default="checkpoints/fantasy-lora", env="ADAPTER_DIR")
    max_new_tokens: int = Field(default=128, env="MAX_NEW_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=0.9, env="TOP_P")
    top_k: int = Field(default=50, env="TOP_K")

    # Data Configuration
    token_dir: Path = Field(default="data/tokenized", env="TOKEN_DIR")
    dataset_limit: int = Field(default=20_000, env="DATASET_LIMIT")
    max_length: int = Field(default=512, env="MAX_LENGTH")

    # Training Configuration
    output_dir: Path = Field(default="checkpoints/fantasy-lora", env="OUTPUT_DIR")
    batch_size: int = Field(default=4, env="BATCH_SIZE")
    gradient_accumulation_steps: int = Field(
        default=4, env="GRADIENT_ACCUMULATION_STEPS"
    )
    num_train_epochs: int = Field(default=3, env="NUM_TRAIN_EPOCHS")
    learning_rate: float = Field(default=2e-4, env="LEARNING_RATE")
    lora_r: int = Field(default=64, env="LORA_R")
    lora_alpha: int = Field(default=128, env="LORA_ALPHA")
    lora_dropout: float = Field(default=0.05, env="LORA_DROPOUT")

    # Weights & Biases
    wandb_project: str = Field(default="fantasy-llm", env="WANDB_PROJECT")
    wandb_entity: Optional[str] = Field(default=None, env="WANDB_ENTITY")
    wandb_api_key: Optional[str] = Field(default=None, env="WANDB_API_KEY")
    wandb_mode: str = Field(default="online", env="WANDB_MODE")

    # Gradio Configuration
    gradio_server_name: str = Field(default="0.0.0.0", env="GRADIO_SERVER_NAME")
    gradio_server_port: int = Field(default=7860, env="GRADIO_SERVER_PORT")
    gradio_share: bool = Field(default=False, env="GRADIO_SHARE")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    api_access_log: bool = Field(default=True, env="API_ACCESS_LOG")

    # Security
    secret_key: str = Field(
        default="dev-secret-key-change-in-production", env="SECRET_KEY"
    )
    api_key: Optional[str] = Field(default=None, env="API_KEY")

    # Database
    database_url: str = Field(default="sqlite:///./fantasy_llm.db", env="DATABASE_URL")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    # CUDA/GPU Configuration
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    torch_home: Path = Field(default="./models/torch_cache", env="TORCH_HOME")

    # HuggingFace Configuration
    hf_home: Path = Field(default="./models/huggingface_cache", env="HF_HOME")
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize settings."""
        super().__init__(**kwargs)
        # Ensure directories exist
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.torch_home.mkdir(parents=True, exist_ok=True)
        self.hf_home.mkdir(parents=True, exist_ok=True)

        # Set environment variables for external libraries
        if self.hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
        os.environ["TORCH_HOME"] = str(self.torch_home)
        os.environ["HF_HOME"] = str(self.hf_home)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices


# Global settings instance
settings = Settings()