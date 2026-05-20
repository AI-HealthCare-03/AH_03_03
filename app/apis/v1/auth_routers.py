from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, status
from fastapi.responses import JSONResponse as Response
from fastapi.security import HTTPAuthorizationCredentials

from app.apis.v1.dependencies import get_firebase_user, optional_security
from app.core import config
from app.core.config import Env
from app.core.firebase import verify_firebase_id_token
from app.dependencies.security import get_request_user
from app.dtos.auth import (
    AvailabilityResponse,
    EmailVerificationSendRequest,
    EmailVerificationSendResponse,
    EmailVerificationVerifyRequest,
    EmailVerificationVerifyResponse,
    FirebaseSyncRequest,
    FirebaseUserResponse,
    LoginRequest,
    LoginResponse,
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    PasswordResetRequestResponse,
    SignUpRequest,
    SimpleMessageResponse,
    TokenRefreshResponse,
)
from app.models.users import User
from app.services import firebase_auth
from app.services.auth import AuthService
from app.services.jwt import JwtService

auth_router = APIRouter(prefix="/auth", tags=["auth"])


def _firebase_user_response(user: User) -> FirebaseUserResponse:
    return FirebaseUserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        nickname=user.nickname,
        role=user.role or "USER",
        is_active=user.is_active,
        auth_provider=user.auth_provider,
        has_firebase_uid=user.firebase_uid is not None,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@auth_router.get("/check-login-id", response_model=AvailabilityResponse, status_code=status.HTTP_200_OK)
async def check_login_id(
    login_id: str,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> AvailabilityResponse:
    return AvailabilityResponse(available=await auth_service.is_login_id_available(login_id))


@auth_router.get("/check-email", response_model=AvailabilityResponse, status_code=status.HTTP_200_OK)
async def check_email(
    email: str,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> AvailabilityResponse:
    return AvailabilityResponse(available=await auth_service.is_email_available(email))


@auth_router.post(
    "/email-verifications/send",
    response_model=EmailVerificationSendResponse,
    status_code=status.HTTP_200_OK,
)
async def send_email_verification_code(
    request: EmailVerificationSendRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> EmailVerificationSendResponse:
    code = await auth_service.send_email_verification_code(request.email)
    # TODO: 운영에서는 debug_code를 응답하지 말고 실제 이메일 발송으로 대체한다.
    return EmailVerificationSendResponse(detail="인증코드가 생성되었습니다.", debug_code=code)


@auth_router.post(
    "/email-verifications/verify",
    response_model=EmailVerificationVerifyResponse,
    status_code=status.HTTP_200_OK,
)
async def verify_email_code(
    request: EmailVerificationVerifyRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> EmailVerificationVerifyResponse:
    return EmailVerificationVerifyResponse(
        verified=await auth_service.verify_email_code(request.email, request.code),
    )


@auth_router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(
    request: SignUpRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> Response:
    await auth_service.signup(request)
    return Response(content={"detail": "회원가입이 성공적으로 완료되었습니다."}, status_code=status.HTTP_201_CREATED)


@auth_router.post("/firebase/sync", response_model=FirebaseUserResponse, status_code=status.HTTP_200_OK)
async def sync_firebase_user(
    request: FirebaseSyncRequest,
    credential: Annotated[HTTPAuthorizationCredentials | None, Depends(optional_security)] = None,
) -> FirebaseUserResponse:
    id_token = request.id_token or (credential.credentials if credential else None)
    if id_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Firebase ID token is missing.")
    try:
        decoded_token = verify_firebase_id_token(id_token)
        user = await firebase_auth.sync_firebase_user(decoded_token)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Firebase ID token.") from exc

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="비활성화된 계정입니다.")
    return _firebase_user_response(user)


@auth_router.get("/firebase/me", response_model=FirebaseUserResponse, status_code=status.HTTP_200_OK)
async def get_firebase_me(user: Annotated[User, Depends(get_firebase_user)]) -> FirebaseUserResponse:
    return _firebase_user_response(user)


@auth_router.post("/login", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def login(
    request: LoginRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> Response:
    user = await auth_service.authenticate(request)
    tokens = await auth_service.login(user)
    resp = Response(
        content=LoginResponse(access_token=str(tokens["access_token"])).model_dump(), status_code=status.HTTP_200_OK
    )
    resp.set_cookie(
        key="refresh_token",
        value=str(tokens["refresh_token"]),
        httponly=True,
        secure=True if config.ENV == Env.PROD else False,
        domain=config.COOKIE_DOMAIN or None,
        expires=tokens["access_token"].payload["exp"],
    )
    return resp


@auth_router.get("/token/refresh", response_model=TokenRefreshResponse, status_code=status.HTTP_200_OK)
async def token_refresh(
    jwt_service: Annotated[JwtService, Depends(JwtService)],
    auth_service: Annotated[AuthService, Depends(AuthService)],
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> Response:
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token is missing.")
    verified_refresh_token = jwt_service.verify_jwt(token=refresh_token, token_type="refresh")
    await auth_service.ensure_refresh_token_active(verified_refresh_token)
    access_token = verified_refresh_token.access_token
    return Response(
        content=TokenRefreshResponse(access_token=str(access_token)).model_dump(), status_code=status.HTTP_200_OK
    )


@auth_router.post(
    "/password-reset/request",
    response_model=PasswordResetRequestResponse,
    status_code=status.HTTP_200_OK,
)
async def request_password_reset(
    request: PasswordResetRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> PasswordResetRequestResponse:
    token = await auth_service.request_password_reset(request.email)
    # TODO: 운영에서는 debug_token을 응답하지 말고 실제 이메일 발송 링크로 대체한다.
    return PasswordResetRequestResponse(detail="비밀번호 재설정 요청이 처리되었습니다.", debug_token=token)


@auth_router.post(
    "/password-reset/confirm",
    response_model=SimpleMessageResponse,
    status_code=status.HTTP_200_OK,
)
async def confirm_password_reset(
    request: PasswordResetConfirmRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> SimpleMessageResponse:
    await auth_service.confirm_password_reset(request)
    return SimpleMessageResponse(detail="비밀번호가 변경되었습니다.")


@auth_router.patch("/password", response_model=SimpleMessageResponse, status_code=status.HTTP_200_OK)
async def change_password(
    request: PasswordChangeRequest,
    user: Annotated[User, Depends(get_request_user)],
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> SimpleMessageResponse:
    await auth_service.change_password(user, request)
    return SimpleMessageResponse(detail="비밀번호가 변경되었습니다.")


@auth_router.post("/logout", response_model=SimpleMessageResponse, status_code=status.HTTP_200_OK)
async def logout(
    auth_service: Annotated[AuthService, Depends(AuthService)],
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> Response:
    await auth_service.logout(refresh_token)
    resp = Response(
        content=SimpleMessageResponse(detail="로그아웃되었습니다.").model_dump(), status_code=status.HTTP_200_OK
    )
    resp.delete_cookie(key="refresh_token", domain=config.COOKIE_DOMAIN or None)
    return resp
