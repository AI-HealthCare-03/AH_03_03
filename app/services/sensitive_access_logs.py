from fastapi import Request

from app.models.logs import SensitiveAccessLog
from app.models.users import User


async def record_sensitive_access(
    *,
    request: Request,
    actor: User,
    target_user_id: int,
    action_type: str,
    resource_type: str,
    resource_id: int | None = None,
    access_reason: str | None = None,
) -> None:
    client_ip = request.client.host if request.client else None
    await SensitiveAccessLog.create(
        request_id=getattr(request.state, "request_id", None),
        actor_user_id=actor.id,
        actor_role=str(actor.role) if actor.role else None,
        target_user_id=target_user_id,
        action_type=action_type,
        resource_type=resource_type,
        resource_id=resource_id,
        access_reason=access_reason,
        method=request.method,
        path=request.url.path,
        client_ip=client_ip,
        user_agent=request.headers.get("user-agent"),
    )


async def safe_record_sensitive_access(
    *,
    request: Request,
    actor: User,
    target_user_id: int,
    action_type: str = "VIEW",
    resource_type: str,
    resource_id: int | None = None,
    access_reason: str | None = None,
) -> None:
    try:
        await record_sensitive_access(
            request=request,
            actor=actor,
            target_user_id=target_user_id,
            action_type=action_type,
            resource_type=resource_type,
            resource_id=resource_id,
            access_reason=access_reason,
        )
    except Exception:
        pass
