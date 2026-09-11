from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_vision_model: str = "llama-3.2-90b-vision-preview"

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "docprocessor"
    postgres_user: str = "docprocessor"
    postgres_password: str = "docprocessor"

    # Redis / Celery
    redis_host: str = "localhost"
    redis_port: int = 6379
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Storage
    upload_dir: str = "./data/uploads"
    processed_dir: str = "./data/processed"

    # Routing thresholds
    auto_approve_confidence_threshold: float = 0.90
    fast_review_confidence_threshold: float = 0.70

    # OCR
    ocr_min_text_density_chars_per_page: int = 40

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def ensure_dirs(self) -> None:
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
