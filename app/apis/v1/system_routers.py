from fastapi import APIRouter

from app.core import config

system_router = APIRouter(prefix="/system", tags=["system"])


@system_router.get("/health")
async def get_system_health():
    return {
        "status": "ok",
        "service": "fastapi",
        "environment": config.ENV,
    }
