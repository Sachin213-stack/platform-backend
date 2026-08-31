import os
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Configuration
    PROJECT_NAME: str = "AI-CTO Backend"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        import json
        try:
            return json.loads(v)
        except Exception:
            return ["*"]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/aicto_db"
    DATABASE_SYNC_URL: str = "postgresql://postgres:postgres@localhost:5432/aicto_db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_STREAM_KEY: str = "telemetry:events:stream"
    REDIS_CONSUMER_GROUP: str = "telemetry_workers"

    # Security & Auth
    # WARNING: Override these in .env for production! Hardcoded defaults are for development only.
    JWT_SECRET_KEY: str = "aicto-super-secret-key-change-in-production-min32chars"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    # Must be a valid Fernet key (32 url-safe base64 bytes). Generate with:
    # python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
    FERNET_SECRET_KEY: str = "aASb7vcf4bLs3HfHDw3xJnen9pKnUf0Nnbwx6ElLAxQ="

    # NVIDIA NIM LLM Endpoints (Priority Fallback Chain)
    NIM_ENDPOINT_1: str = "https://integrate.api.nvidia.com/v1"
    NIM_API_KEY_1: str = ""
    NIM_MODEL_1: str = "meta/llama-3.1-70b-instruct"

    NIM_ENDPOINT_2: str = "https://integrate.api.nvidia.com/v1"
    NIM_API_KEY_2: str = ""
    NIM_MODEL_2: str = "mistralai/mixtral-8x22b-instruct-v0.1"

    NIM_ENDPOINT_3: str = "https://integrate.api.nvidia.com/v1"
    NIM_API_KEY_3: str = ""
    NIM_MODEL_3: str = "meta/llama-3.1-8b-instruct"

    # Observability
    SENTRY_DSN: str = ""


settings = Settings()
