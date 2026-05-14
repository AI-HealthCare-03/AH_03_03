"""
ai_worker/vision/settings.py
환경변수 기반 설정.
envs/.local.env 에 아래 항목을 추가하세요:

    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class VisionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    cv_max_image_size_mb: int = 10
    cv_min_confidence: float = 0.6
