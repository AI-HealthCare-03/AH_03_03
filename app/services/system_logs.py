import traceback

from fastapi import Request

from app.core import config
from app.models.logs import SystemErrorLog


async def create_system_error_log(request: Request, exc: Exception, status_code: int = 500) -> None:
    stack_trace = None if config.is_production else traceback.format_exc()
    client_ip = request.client.host if request.client else None

    await SystemErrorLog.create(
        request_id=getattr(request.state, "request_id", None),
        user_id=getattr(request.state, "user_id", None),
        method=request.method,
        path=request.url.path,
        status_code=status_code,
        error_type=exc.__class__.__name__,
        error_message=str(exc) or None,
        stack_trace=stack_trace,
        client_ip=client_ip,
        user_agent=request.headers.get("user-agent"),
    )
