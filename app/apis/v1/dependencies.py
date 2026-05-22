from typing import Annotated, Any

from fastapi import Depends, HTTPException, status

from app.dependencies.security import get_request_user
from app.models.users import User, UserRole

MONITOR_ROLES = {UserRole.MONITOR, UserRole.OPERATOR, UserRole.ADMIN, UserRole.SUPER_ADMIN}
OPERATOR_ROLES = {UserRole.OPERATOR, UserRole.ADMIN, UserRole.SUPER_ADMIN}
ADMIN_ROLES = {UserRole.ADMIN, UserRole.SUPER_ADMIN}
SUPER_ADMIN_ROLES = {UserRole.SUPER_ADMIN}


def raise_not_found(message: str) -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)


def raise_forbidden(message: str) -> None:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=message)


def ensure_admin_user(current_user: User) -> None:
    if is_admin_user(current_user):
        return
    raise_forbidden("관리자 권한이 필요합니다.")


def ensure_super_admin_user(current_user: User) -> None:
    if is_super_admin_user(current_user):
        return
    raise_forbidden("관리자 권한이 필요합니다.")


def ensure_monitor_user(current_user: User) -> None:
    if is_monitor_user(current_user):
        return
    raise_forbidden("관리자 권한이 필요합니다.")


def ensure_operator_user(current_user: User) -> None:
    if is_operator_user(current_user):
        return
    raise_forbidden("관리자 권한이 필요합니다.")


def normalize_user_role(current_user: User | None) -> str:
    if current_user is None:
        return ""
    return str(getattr(current_user, "role", "") or "").upper()


def _user_role(current_user: User | None) -> UserRole | None:
    if current_user is None:
        return None
    try:
        return UserRole(normalize_user_role(current_user))
    except ValueError:
        return None


def is_monitor_user(current_user: User) -> bool:
    return _user_role(current_user) in MONITOR_ROLES


def is_operator_user(current_user: User) -> bool:
    return _user_role(current_user) in OPERATOR_ROLES


def is_admin_user(current_user: User) -> bool:
    # users.role is the source of truth for authorization.
    # is_admin remains on the User model only for legacy data compatibility.
    return _user_role(current_user) in ADMIN_ROLES


def is_super_admin_user(current_user: User) -> bool:
    return _user_role(current_user) in SUPER_ADMIN_ROLES


def ensure_owner(resource_user_id: int | None, current_user: User) -> None:
    if resource_user_id is None or int(resource_user_id) != int(current_user.id):
        raise_forbidden("해당 리소스에 접근할 권한이 없습니다.")


def ensure_owner_or_admin(resource_user_id: int | None, current_user: User) -> None:
    if is_admin_user(current_user):
        return
    ensure_owner(resource_user_id, current_user)


def ensure_owner_or_operator(resource_user_id: int | None, current_user: User) -> None:
    if is_operator_user(current_user):
        return
    ensure_owner(resource_user_id, current_user)


def ensure_found(resource: Any | None, message: str) -> Any:
    if resource is None:
        raise_not_found(message)
    return resource


async def require_admin_user(user: Annotated[User, Depends(get_request_user)]) -> User:
    ensure_admin_user(user)
    return user


async def require_super_admin_user(user: Annotated[User, Depends(get_request_user)]) -> User:
    ensure_super_admin_user(user)
    return user


async def require_monitor_user(user: Annotated[User, Depends(get_request_user)]) -> User:
    ensure_monitor_user(user)
    return user


async def require_operator_user(user: Annotated[User, Depends(get_request_user)]) -> User:
    ensure_operator_user(user)
    return user
