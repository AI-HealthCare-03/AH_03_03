from fastapi import FastAPI
from tortoise import Tortoise
from tortoise.contrib.fastapi import register_tortoise

from app.core import config

TORTOISE_APP_MODELS = [
    "aerich.models",
    "app.models.analysis",
    "app.models.async_jobs",
    "app.models.challenges",
    "app.models.diets",
    "app.models.exams",
    "app.models.faqs",
    "app.models.family",
    "app.models.health",
    "app.models.llm_logs",
    "app.models.logs",
    "app.models.medications",
    "app.models.notifications",
    "app.models.rag",
    "app.models.settings",
    "app.models.users",
]

TORTOISE_ORM = {
    "connections": {
        "default": {
            "engine": "tortoise.backends.asyncpg",
            "credentials": {
                "host": config.DB_HOST,
                "port": config.DB_PORT,
                "user": config.DB_USER,
                "password": config.DB_PASSWORD,
                "database": config.DB_NAME,
                "minsize": config.DB_POOL_MIN_SIZE,
                "maxsize": config.db_pool_max_size,
                "command_timeout": config.DB_COMMAND_TIMEOUT,
            },
        },
    },
    "apps": {
        "models": {
            "models": TORTOISE_APP_MODELS,
        },
    },
    "timezone": "Asia/Seoul",
}


def initialize_tortoise(app: FastAPI) -> None:
    Tortoise.init_models(TORTOISE_APP_MODELS, "models")
    register_tortoise(app, config=TORTOISE_ORM)
