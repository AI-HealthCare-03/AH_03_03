from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, status
from fastapi.responses import JSONResponse as Response

from app.core import config
from app.dependencies.security import get_request_user
from app.dtos.auth import (
    AvailabilityResponse,
    EmailVerificationSendRequest,
    EmailVerificationSendResponse,
    EmailVerificationVerifyRequest,
    EmailVerificationVerifyResponse,
    FindLoginIdRequest,
    FindLoginIdResponse,
    LoginRequest,
    LoginResponse,
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    PasswordResetRequestResponse,
    PhoneVerificationSendRequest,
    PhoneVerificationSendResponse,
    PhoneVerificationVerifyRequest,
    PhoneVerificationVerifyResponse,
    SignUpRequest,
    SimpleMessageResponse,
    TokenRefreshResponse,
)
from app.models.users import User
from app.services.auth import AuthService
from app.services.jwt import JwtService

auth_router = APIRouter(prefix="/auth", tags=["auth"])


def _refresh_cookie_settings() -> dict[str, object]:
    return {
        "key": config.REFRESH_TOKEN_COOKIE_NAME,
        "path": config.REFRESH_TOKEN_COOKIE_PATH,
        "domain": config.refresh_token_cookie_domain,
        "secure": config.refresh_token_cookie_secure,
        "httponly": True,
        "samesite": config.REFRESH_TOKEN_COOKIE_SAMESITE,
    }


def _refresh_cookie_max_age(refresh_token_exp: int) -> int:
    expires_at = datetime.fromtimestamp(refresh_token_exp, tz=config.TIMEZONE)
    return max(0, int((expires_at - datetime.now(config.TIMEZONE)).total_seconds()))


def _refresh_cookie_expires(refresh_token_exp: int) -> datetime:
    return datetime.fromtimestamp(refresh_token_exp, tz=UTC)


def _allow_auth_debug_response(debug_enabled: bool) -> bool:
    # 인증코드/토큰 디버그 응답은 시연 편의를 위한 local/demo 전용 장치다.
    return debug_enabled and not config.is_production


@auth_router.get("/check-login-id", response_model=AvailabilityResponse, status_code=status.HTTP_200_OK)
async def check_login_id(
    login_id: str,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> AvailabilityResponse:
    available = await auth_service.is_login_id_available(login_id)
    return AvailabilityResponse(
        available=available,
        message="사용 가능한 아이디입니다." if available else "이미 사용 중인 아이디입니다.",
    )


@auth_router.get("/check-email", response_model=AvailabilityResponse, status_code=status.HTTP_200_OK)
async def check_email(
    email: str,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> AvailabilityResponse:
    available = await auth_service.is_email_available(email.strip().lower())
    return AvailabilityResponse(
        available=available,
        message="사용 가능한 이메일입니다." if available else "이미 사용 중인 이메일입니다.",
    )


@auth_router.get("/check-phone", response_model=AvailabilityResponse, status_code=status.HTTP_200_OK)
async def check_phone(
    phone_number: str,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> AvailabilityResponse:
    available = await auth_service.is_phone_number_available(phone_number)
    return AvailabilityResponse(
        available=available,
        message="사용 가능한 휴대폰 번호입니다." if available else "이미 사용 중인 휴대폰 번호입니다.",
    )


@auth_router.post("/find-login-id", response_model=FindLoginIdResponse, status_code=status.HTTP_200_OK)
async def find_login_id(
    request: FindLoginIdRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> FindLoginIdResponse:
    return await auth_service.find_login_id(request)


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
    return EmailVerificationSendResponse(
        detail="인증코드가 생성되었습니다.",
        debug_code=code if _allow_auth_debug_response(config.EMAIL_VERIFICATION_DEBUG) else None,
    )


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


@auth_router.post(
    "/phone-verifications/send",
    response_model=PhoneVerificationSendResponse,
    status_code=status.HTTP_200_OK,
    deprecated=True,
    include_in_schema=False,
)
async def send_phone_verification_code(
    request: PhoneVerificationSendRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> PhoneVerificationSendResponse:
    # 기존 클라이언트 호환을 위해 endpoint는 남기되 MVP 공식 인증 경로에서는 제외한다.
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="휴대폰 인증은 현재 MVP 범위에서 제공하지 않습니다. 이메일 인증을 사용해주세요.",
    )


@auth_router.post(
    "/phone-verifications/verify",
    response_model=PhoneVerificationVerifyResponse,
    status_code=status.HTTP_200_OK,
    deprecated=True,
    include_in_schema=False,
)
async def verify_phone_code(
    request: PhoneVerificationVerifyRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> PhoneVerificationVerifyResponse:
    # 404보다 410을 반환해 호출자가 정책적으로 중단된 기능임을 알 수 있게 한다.
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="휴대폰 인증은 현재 MVP 범위에서 제공하지 않습니다. 이메일 인증을 사용해주세요.",
    )


@auth_router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(
    request: SignUpRequest,
    auth_service: Annotated[AuthService, Depends(AuthService)],
) -> Response:
    await auth_service.signup(request)
    return Response(content={"detail": "회원가입이 성공적으로 완료되었습니다."}, status_code=status.HTTP_201_CREATED)


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
        **_refresh_cookie_settings(),
        value=str(tokens["refresh_token"]),
        max_age=_refresh_cookie_max_age(tokens["refresh_token"].payload["exp"]),
        expires=_refresh_cookie_expires(tokens["refresh_token"].payload["exp"]),
    )
    return resp


@auth_router.get("/token/refresh", response_model=TokenRefreshResponse, status_code=status.HTTP_200_OK)
async def token_refresh(
    jwt_service: Annotated[JwtService, Depends(JwtService)],
    auth_service: Annotated[AuthService, Depends(AuthService)],
    refresh_token: Annotated[str | None, Cookie(alias=config.REFRESH_TOKEN_COOKIE_NAME)] = None,
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
    return PasswordResetRequestResponse(
        detail="입력한 이메일로 비밀번호 재설정 링크를 전송했습니다.",
        debug_token=token if _allow_auth_debug_response(config.PASSWORD_RESET_DEBUG) else None,
    )


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
    refresh_token: Annotated[str | None, Cookie(alias=config.REFRESH_TOKEN_COOKIE_NAME)] = None,
) -> Response:
    await auth_service.logout(refresh_token)
    resp = Response(
        content=SimpleMessageResponse(detail="로그아웃되었습니다.").model_dump(), status_code=status.HTTP_200_OK
    )
    resp.delete_cookie(**_refresh_cookie_settings())
    return resp
