import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from tortoise import Tortoise

from app.apis.v1 import v1_routers
from app.core import config
from app.core.db.databases import initialize_tortoise
from app.services.system_logs import create_system_error_log


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    _ = app_instance
    try:
        yield
    finally:
        if Tortoise._inited:
            await Tortoise.close_connections()


app = FastAPI(
    default_response_class=ORJSONResponse,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception as exc:
        try:
            await create_system_error_log(request, exc)
        except Exception:
            pass
        return ORJSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
    response.headers["X-Request-ID"] = request_id
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
initialize_tortoise(app)

app.include_router(v1_routers)
