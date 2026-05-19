from typing import Any

from fastapi import HTTPException, status

from app.models.users import User


def raise_not_found(message: str) -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)


def raise_forbidden(message: str) -> None:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=message)


def ensure_admin_user(current_user: User) -> None:
    if is_admin_user(current_user):
        return
    raise_forbidden("관리자 권한이 필요합니다.")


def is_admin_user(current_user: User) -> bool:
    role = str(getattr(current_user, "role", "") or "").lower()
    return bool(getattr(current_user, "is_admin", False) or role == "admin")


def ensure_owner(resource_user_id: int | None, current_user: User) -> None:
    if resource_user_id is None or int(resource_user_id) != int(current_user.id):
        raise_forbidden("해당 리소스에 접근할 권한이 없습니다.")


def ensure_owner_or_admin(resource_user_id: int | None, current_user: User) -> None:
    if is_admin_user(current_user):
        return
    ensure_owner(resource_user_id, current_user)


def ensure_found(resource: Any | None, message: str) -> Any:
    if resource is None:
        raise_not_found(message)
    return resource
