from fastapi import APIRouter
from redis.asyncio import Redis
from tortoise import Tortoise

from app.core import config
from app.services.email_service import EmailService

system_router = APIRouter(prefix="/system", tags=["system"])


@system_router.get("/health")
async def get_system_health():
    checks = {
        "api": "ok",
        "database": "unknown",
        "redis": "unknown",
        "request_id": "ok",
        "system_error_logging": "ok",
        "email_service": EmailService().status(),
        "twilio_verify": config.twilio_verify_status,
    }
    details: dict[str, str] = {}

    try:
        connection = Tortoise.get_connection("default")
        await connection.execute_query("SELECT 1")
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = "error"
        details["database"] = str(exc)

    if not config.REDIS_HOST:
        checks["redis"] = "not_configured"
    else:
        redis_client = Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            socket_connect_timeout=1,
            socket_timeout=1,
            decode_responses=True,
        )
        try:
            pong = await redis_client.ping()
            checks["redis"] = "ok" if pong else "error"
        except Exception as exc:
            checks["redis"] = "degraded"
            details["redis"] = str(exc)
        finally:
            await redis_client.aclose()

    if checks["database"] == "error":
        overall_status = "error"
    elif any(value in {"degraded", "error", "not_configured", "misconfigured"} for value in checks.values()):
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return {
        "status": overall_status,
        "service": "fastapi",
        "environment": config.ENV,
        "checks": checks,
        "details": details,
    }
