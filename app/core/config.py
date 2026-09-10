import os
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator, model_validator
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
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://platform-3aya.onrender.com",
    ]

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

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_database_url(cls, v: str) -> str:
        if isinstance(v, str) and v:
            if v.startswith("postgres://"):
                v = v.replace("postgres://", "postgresql+asyncpg://", 1)
            elif v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
                v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
            if "sslmode=" in v:
                v = v.replace("sslmode=require", "ssl=require").replace("sslmode=prefer", "ssl=prefer").replace("sslmode=disable", "ssl=disable")
            if "ssl=true" in v.lower():
                v = v.replace("ssl=true", "ssl=require").replace("ssl=True", "ssl=require")
            # For remote database hosts (e.g. Render Managed Postgres), enforce ssl=require if not explicitly specified
            if "@" in v and not any(h in v.split("@")[-1] for h in ["localhost", "127.0.0.1", "postgres:5432", "postgres/"]):
                if "ssl=" not in v and "sslmode=" not in v:
                    separator = "&" if "?" in v else "?"
                    v = f"{v}{separator}ssl=require"
        return v

    @field_validator("DATABASE_SYNC_URL", mode="before")
    @classmethod
    def assemble_database_sync_url(cls, v: str) -> str:
        if isinstance(v, str) and v:
            if v.startswith("postgres://"):
                v = v.replace("postgres://", "postgresql://", 1)
            if "@" in v and not any(h in v.split("@")[-1] for h in ["localhost", "127.0.0.1", "postgres:5432", "postgres/"]):
                if "sslmode=" not in v and "ssl=" not in v:
                    separator = "&" if "?" in v else "?"
                    v = f"{v}{separator}sslmode=require"
        return v

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

    # Kimi (Moonshot AI) LLM Configuration (Sole LLM Provider)
    KIMI_API_KEY: str = ""
    MOONSHOT_API_KEY: str = ""
    KIMI_BASE_URL: str = "https://api.moonshot.ai/v1"
    KIMI_MODEL_PRIMARY: str = "kimi-k3"
    KIMI_MODEL_SECONDARY: str = "kimi-k2.6"
    KIMI_MODEL_FALLBACK: str = "moonshot-v1-128k"
    KIMI_TIMEOUT_SECONDS: float = 30.0

    @property
    def effective_kimi_api_key(self) -> str:
        """Returns the configured Kimi API key from KIMI_API_KEY, MOONSHOT_API_KEY, or env."""
        key = (
            self.KIMI_API_KEY
            or self.MOONSHOT_API_KEY
            or os.getenv("KIMI_API_KEY", "")
            or os.getenv("MOONSHOT_API_KEY", "")
        )
        return (key or "").strip('"\'').strip()

    # Observability & Logging
    SENTRY_DSN: str = ""
    LOG_LEVEL: str = "DEBUG" if os.getenv("ENVIRONMENT") == "development" else "INFO"
    LOG_FORMAT: str = "auto"  # 'auto' (console for dev, json for prod), 'json', or 'console'
    LOG_FILE_PATH: str = "logs/aicto.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB per file
    LOG_BACKUP_COUNT: int = 5  # Keep 5 rotated backup files

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Enforces that real, secure credentials are supplied in production environments."""
        if self.ENVIRONMENT == "production":
            insecure_jwt_defaults = {
                "aicto-super-secret-key-change-in-production-min32chars",
                "change-this-super-secret-key-in-production-min-32-chars-long",
                "secret",
                "",
            }
            if self.JWT_SECRET_KEY in insecure_jwt_defaults or len(self.JWT_SECRET_KEY) < 32:
                raise ValueError(
                    "Insecure or missing JWT_SECRET_KEY in production! "
                    "You must configure a unique, high-entropy secret (minimum 32 characters) in Render's environment variables."
                )

            insecure_fernet_defaults = {
                "aASb7vcf4bLs3HfHDw3xJnen9pKnUf0Nnbwx6ElLAxQ=",
                "ZI4lUaSB3LjrE4LrN25nmQvx98H5pc4Q29pprotD6EA=",
                "",
            }
            if self.FERNET_SECRET_KEY in insecure_fernet_defaults:
                raise ValueError(
                    "Insecure or default FERNET_SECRET_KEY in production! "
                    "You must generate and configure a unique FERNET_SECRET_KEY in Render's environment variables. "
                    "Generate via: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
                )
        return self


settings = Settings()
