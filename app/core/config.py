import os
import uuid
import zoneinfo
from dataclasses import field
from enum import StrEnum
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    PROD = "prod"
    PRODUCTION = "production"


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    ENV: Env = Env.LOCAL
    SECRET_KEY: str = f"default-secret-key{uuid.uuid4().hex}"
    TIMEZONE: zoneinfo.ZoneInfo = field(default_factory=lambda: zoneinfo.ZoneInfo("Asia/Seoul"))
    TEMPLATE_DIR: str = os.path.join(Path(__file__).resolve().parent.parent, "templates")

    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = "pw1234"
    DB_NAME: str = "ai_health"
    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 5
    DB_COMMAND_TIMEOUT: int = 60
    DB_CONNECTION_POOL_MAXSIZE: int | None = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    SCHEDULER_ENABLED: bool = False
    SCHEDULER_INTERVAL_SECONDS: int = 60
    UPLOAD_STORAGE_DIR: str = "var/uploads"
    STORAGE_BACKEND: str = "local"
    LOCAL_STORAGE_ROOT: str = "var/storage"
    S3_BUCKET_NAME: str | None = None
    S3_REGION: str = "ap-northeast-2"
    S3_PREFIX: str = ""
    S3_PRESIGNED_URL_EXPIRES_SECONDS: int = 3600
    LOGIN_FAILURE_LIMIT: int = 5
    LOGIN_SOFT_LOCK_MINUTES: int = 1
    ACCOUNT_LOCK_MINUTES: int = 15
    # local/demo에서는 이메일 인증 흐름 검증을 돕되, prod에서는 응답에 인증값이 노출되면 안 된다.
    EMAIL_ENABLED: bool = False
    EMAIL_VERIFICATION_DEBUG: bool = False
    PASSWORD_RESET_DEBUG: bool = False
    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_FROM_EMAIL: str | None = None
    SMTP_FROM_NAME: str = "AI HealthCare"
    SMTP_USE_TLS: bool = True
    FRONTEND_BASE_URL: str = "http://localhost:5173"
    # OCR/CV 외부 provider는 비용/안정성 때문에 명시 플래그가 켜진 경우에만 공식 경로에서 사용한다.
    GPT_VISION_FALLBACK_ENABLED: bool = False
    FOOD_CV_CONFIDENCE_THRESHOLD: float = 0.75
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    DIET_GPT_VISION_ENABLED: bool = False
    DIET_GPT_VISION_MODEL: str = "gpt-4o-mini"
    DIET_VISION_PROVIDER: str = "rule_based"
    DIET_MFDS_ENABLED: bool = False
    MFDS_SERVICE_KEY: str | None = None
    MFDS_SERVICE_KEY_ENCODED: str | None = None
    DIET_MFDS_TIMEOUT_SECONDS: float = 5.0
    DIET_MFDS_MAX_CANDIDATES: int = 5
    EXAM_OCR_PROVIDER: str = "auto"
    EXAM_GPT_VISION_ENABLED: bool = False
    EXAM_GPT_VISION_MODEL: str = "gpt-4o-mini"
    PADDLE_OCR_ENABLED: bool = False
    MEDICATION_OCR_PROVIDER: str = "fallback"
    MEDICATION_GPT_VISION_ENABLED: bool = False
    MEDICATION_GPT_VISION_MODEL: str = "gpt-4o-mini"
    CHATBOT_USE_REAL_LLM: bool = False
    RAG_ENABLED: bool = False
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_BASE_URL: str | None = None
    LANGFUSE_HOST: str | None = None
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None

    COOKIE_DOMAIN: str = "localhost"
    REFRESH_TOKEN_COOKIE_NAME: str = "refresh_token"
    REFRESH_TOKEN_COOKIE_PATH: str = "/api/v1/auth"
    REFRESH_TOKEN_COOKIE_SAMESITE: str = "lax"
    REFRESH_TOKEN_COOKIE_SECURE: bool | None = None
    CORS_ALLOW_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 14 * 24 * 60
    JWT_LEEWAY: int = 5

    @model_validator(mode="after")
    def validate_production_security_settings(self) -> "Config":
        if not self.is_production:
            return self

        secret_key = self.SECRET_KEY.strip()
        if not secret_key or secret_key == "PLEASE_CHANGE_ME" or secret_key.startswith("default-secret-key"):
            raise ValueError("SECRET_KEY must be set to a strong non-default value in production.")

        if not self.refresh_token_cookie_secure:
            raise ValueError("REFRESH_TOKEN_COOKIE_SECURE must be true in production.")

        cookie_domain = self.COOKIE_DOMAIN.strip().lower()
        if cookie_domain in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("COOKIE_DOMAIN must not point to localhost in production.")

        same_site = self.REFRESH_TOKEN_COOKIE_SAMESITE.strip().lower()
        if same_site not in {"lax", "strict", "none"}:
            raise ValueError("REFRESH_TOKEN_COOKIE_SAMESITE must be one of: lax, strict, none.")

        return self

    @property
    def cors_allow_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ALLOW_ORIGINS.split(",") if origin.strip()]

    @property
    def db_pool_max_size(self) -> int:
        return self.DB_CONNECTION_POOL_MAXSIZE or self.DB_POOL_MAX_SIZE

    @property
    def is_production(self) -> bool:
        return self.ENV in {Env.PROD, Env.PRODUCTION}

    @property
    def refresh_token_cookie_secure(self) -> bool:
        if self.REFRESH_TOKEN_COOKIE_SECURE is not None:
            return self.REFRESH_TOKEN_COOKIE_SECURE
        return self.is_production

    @property
    def refresh_token_cookie_domain(self) -> str | None:
        if not self.COOKIE_DOMAIN:
            return None
        if not self.is_production and self.COOKIE_DOMAIN == "localhost":
            return None
        return self.COOKIE_DOMAIN
