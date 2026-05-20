from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.firebase import verify_firebase_id_token
from app.models.users import User
from app.repositories.user_repository import UserRepository
from app.services.firebase_auth import sync_firebase_user
from app.services.jwt import JwtService

optional_security = HTTPBearer(auto_error=False)


def raise_not_found(message: str) -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)


def raise_forbidden(message: str) -> None:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=message)


def ensure_admin_user(current_user: User) -> None:
    if is_admin_user(current_user):
        return
    raise_forbidden("관리자 권한이 필요합니다.")


def is_admin_user(current_user: User) -> bool:
    if current_user is None:
        return False

    # TODO: 신규 권한 로직은 role을 진실 공급원으로 사용하고, is_admin은 legacy 호환용으로만 유지한다.
    normalized_role = str(getattr(current_user, "role", "") or "").upper()
    if normalized_role in {"ADMIN", "ROLE_ADMIN"}:
        return True
    return bool(getattr(current_user, "is_admin", False))


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


async def get_firebase_user(
    credential: Annotated[HTTPAuthorizationCredentials | None, Depends(optional_security)],
) -> User:
    if credential is None or credential.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Firebase ID token is missing.")
    try:
        decoded_token = verify_firebase_id_token(credential.credentials)
        user = await sync_firebase_user(decoded_token)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Firebase ID token.") from exc

    if not user.is_active:
        raise_forbidden("비활성화된 계정입니다.")
    return user


async def get_optional_firebase_user(
    credential: Annotated[HTTPAuthorizationCredentials | None, Depends(optional_security)],
) -> User | None:
    if credential is None:
        return None
    try:
        return await get_firebase_user(credential)
    except HTTPException:
        return None


async def get_request_user_with_firebase(
    credential: Annotated[HTTPAuthorizationCredentials | None, Depends(optional_security)],
) -> User:
    if credential is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization token is missing.")

    try:
        return await get_firebase_user(credential)
    except HTTPException:
        # Firebase is an optional provider for the MVP frontend. If Firebase
        # credentials are missing or the bearer token is not a Firebase ID token,
        # continue with the existing FastAPI JWT authentication path.
        pass

    try:
        verified = JwtService().verify_jwt(token=credential.credentials, token_type="access")
        user = await UserRepository().get_user(verified.payload["user_id"])
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticate Failed.") from exc

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticate Failed.")
    if not user.is_active:
        raise_forbidden("비활성화된 계정입니다.")
    return user
