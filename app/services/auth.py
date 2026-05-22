import asyncio
import hashlib
import json
import secrets
import urllib.parse
import urllib.request
from base64 import b64encode
from datetime import datetime, timedelta
from random import randint

from fastapi.exceptions import HTTPException
from pydantic import EmailStr
from starlette import status
from tortoise.transactions import in_transaction

from app.core import config
from app.core.jwt.tokens import AccessToken
from app.core.jwt.tokens import RefreshToken as JWTRefreshToken
from app.core.utils.common import normalize_phone_number
from app.core.utils.security import hash_password, verify_password
from app.dtos.auth import (
    FindLoginIdRequest,
    FindLoginIdResponse,
    LoginRequest,
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    SignUpRequest,
)
from app.dtos.settings import UserSettingCreateRequest
from app.models.users import User
from app.repositories import setting_repository
from app.repositories.user_repository import UserRepository
from app.services.jwt import JwtService

VERIFICATION_CODE_TTL_MINUTES = 10
PASSWORD_RESET_TOKEN_TTL_MINUTES = 30
INVALID_LOGIN_MESSAGE = "아이디 또는 비밀번호가 올바르지 않습니다."
ACCOUNT_LOCKED_MESSAGE = "로그인 시도가 여러 번 실패했습니다. 잠시 후 다시 시도하거나 추가 확인을 진행해주세요."
PHONE_VERIFICATION_PURPOSE = "PHONE_VERIFICATION"


class AuthService:
    def __init__(self):
        self.user_repo = UserRepository()
        self.jwt_service = JwtService()

    async def signup(self, data: SignUpRequest) -> User:
        login_id = data.login_id or self._default_login_id(str(data.email))
        await self.check_login_id_exists(login_id)

        # 이메일 중복 체크
        await self.check_email_exists(data.email)

        # 입력받은 휴대폰 번호를 노말라이즈
        normalized_phone_number = normalize_phone_number(data.phone_number)

        # 휴대폰 번호 중복 체크
        await self.check_phone_number_exists(normalized_phone_number)

        # 유저 생성
        async with in_transaction():
            user = await self.user_repo.create_user(
                login_id=login_id,
                email=data.email,
                hashed_password=hash_password(data.password),  # 해시화된 비밀번호를 사용
                name=data.name,
                nickname=data.nickname,
                phone_number=normalized_phone_number,
                gender=data.gender,
                birthday=data.birth_date,
                address=data.address,
                profile_image_url=data.profile_image_url,
            )
            await self.user_repo.create_user_consent(
                user_id=user.id,
                sensitive_data_agreed=data.sensitive_data_agreed,
                marketing_agreed=data.marketing_agreed,
            )
            if await setting_repository.get_user_setting_by_user(user.id) is None:
                await setting_repository.create_user_setting(user.id, UserSettingCreateRequest().model_dump())

            return user

    async def authenticate(self, data: LoginRequest) -> User:
        if data.login_id is not None:
            user = await self.user_repo.get_user_by_login_id(data.login_id)
        else:
            user = await self.user_repo.get_user_by_email(str(data.email))
        if not user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_LOGIN_MESSAGE)

        now = datetime.now(config.TIMEZONE)
        if user.locked_until is not None and user.locked_until > now:
            raise HTTPException(status_code=status.HTTP_423_LOCKED, detail=ACCOUNT_LOCKED_MESSAGE)

        # 비밀번호 검증
        if not verify_password(data.password, user.hashed_password):
            await self._record_failed_login(user, now)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_LOGIN_MESSAGE)

        # 활성 사용자 체크
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_423_LOCKED, detail="비활성화된 계정입니다.")

        await self.user_repo.reset_login_failures(user.id)
        user.failed_login_count = 0
        user.locked_until = None
        return user

    async def login(self, user: User) -> dict[str, AccessToken | JWTRefreshToken]:
        await self.user_repo.update_last_login(user.id)
        tokens = self.jwt_service.issue_jwt_pair(user)
        refresh_token = tokens["refresh_token"]
        await self.user_repo.create_refresh_token(
            user_id=user.id,
            token_jti=refresh_token.payload["jti"],
            expires_at=datetime.fromtimestamp(refresh_token.payload["exp"], tz=config.TIMEZONE),
        )
        return tokens

    async def check_email_exists(self, email: str | EmailStr) -> None:
        if await self.user_repo.exists_by_email(email):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="이미 사용중인 이메일입니다.")

    async def check_login_id_exists(self, login_id: str) -> None:
        if await self.user_repo.exists_by_login_id(login_id):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="이미 사용중인 아이디입니다.")

    async def check_phone_number_exists(self, phone_number: str) -> None:
        if await self.user_repo.exists_by_phone_number(phone_number):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="이미 사용중인 휴대폰 번호입니다.")

    async def is_login_id_available(self, login_id: str) -> bool:
        return not await self.user_repo.exists_by_login_id(login_id)

    async def is_email_available(self, email: str | EmailStr) -> bool:
        return not await self.user_repo.exists_by_email(email)

    async def is_phone_number_available(self, phone_number: str) -> bool:
        return not await self.user_repo.exists_by_phone_number(normalize_phone_number(phone_number))

    async def find_login_id(self, data: FindLoginIdRequest) -> FindLoginIdResponse:
        name = data.name.strip()
        user = None
        if data.email is not None:
            user = await self.user_repo.get_user_by_name_and_email(name, str(data.email).strip().lower())
        if user is None and data.phone_number:
            user = await self.user_repo.get_user_by_name_and_phone_number(
                name,
                normalize_phone_number(data.phone_number),
            )

        if user is None or not user.login_id:
            return FindLoginIdResponse(
                found=False,
                masked_login_id=None,
                message="일치하는 계정을 찾을 수 없습니다.",
            )

        return FindLoginIdResponse(
            found=True,
            masked_login_id=self._mask_login_id(user.login_id),
            message="가입된 아이디를 찾았습니다.",
        )

    async def send_email_verification_code(self, email: str | EmailStr) -> str:
        code = f"{randint(0, 999999):06d}"
        expires_at = datetime.now(config.TIMEZONE) + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)
        await self.user_repo.create_verification_code(
            email=str(email),
            code_hash=self._digest(code),
            expires_at=expires_at,
        )
        return code

    async def send_phone_verification_code(self, phone_number: str) -> str | None:
        normalized_phone_number = normalize_phone_number(phone_number)
        if config.TWILIO_ENABLED:
            await self._send_twilio_verification(normalized_phone_number)
            return None

        code = f"{randint(0, 999999):06d}"
        expires_at = datetime.now(config.TIMEZONE) + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)
        await self.user_repo.create_verification_code(
            email=self._phone_verification_key(normalized_phone_number),
            code_hash=self._digest(code),
            expires_at=expires_at,
            purpose=PHONE_VERIFICATION_PURPOSE,
        )
        return code

    async def verify_email_code(self, email: str | EmailStr, code: str) -> bool:
        verification = await self.user_repo.get_latest_verification_code(str(email))
        now = datetime.now(config.TIMEZONE)
        if verification is None or verification.expires_at < now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="인증코드가 만료되었거나 존재하지 않습니다."
            )
        if verification.code_hash != self._digest(code):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="인증코드가 올바르지 않습니다.")

        await self.user_repo.mark_verification_code_used(verification, now)
        user = await self.user_repo.get_user_by_email(str(email))
        if user is not None:
            await self.user_repo.update_user_fields(user.id, {"email_verified_at": now})
        return True

    async def verify_phone_code(self, phone_number: str, code: str) -> bool:
        normalized_phone_number = normalize_phone_number(phone_number)
        if config.TWILIO_ENABLED:
            return await self._check_twilio_verification(normalized_phone_number, code)

        verification = await self.user_repo.get_latest_verification_code(
            self._phone_verification_key(normalized_phone_number),
            PHONE_VERIFICATION_PURPOSE,
        )
        now = datetime.now(config.TIMEZONE)
        if verification is None or verification.expires_at < now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="인증번호가 만료되었거나 존재하지 않습니다."
            )
        if verification.code_hash != self._digest(code):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="인증번호가 올바르지 않습니다.")

        await self.user_repo.mark_verification_code_used(verification, now)
        return True

    async def request_password_reset(self, email: str | EmailStr) -> str:
        user = await self.user_repo.get_user_by_email(str(email))
        token = secrets.token_urlsafe(32)
        if user is not None:
            expires_at = datetime.now(config.TIMEZONE) + timedelta(minutes=PASSWORD_RESET_TOKEN_TTL_MINUTES)
            await self.user_repo.create_password_reset_token(
                user_id=user.id,
                token_hash=self._digest(token),
                expires_at=expires_at,
            )
        return token

    async def confirm_password_reset(self, data: PasswordResetConfirmRequest) -> None:
        token = await self.user_repo.get_password_reset_token(self._digest(data.token))
        now = datetime.now(config.TIMEZONE)
        if token is None or token.expires_at < now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="비밀번호 재설정 토큰이 유효하지 않습니다."
            )
        await self.user_repo.update_password(token.user_id, hash_password(data.new_password))
        await self.user_repo.mark_password_reset_token_used(token, now)

    async def change_password(self, user: User, data: PasswordChangeRequest) -> None:
        if not verify_password(data.current_password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="현재 비밀번호가 올바르지 않습니다.")
        await self.user_repo.update_password(user.id, hash_password(data.new_password))

    async def logout(self, refresh_token: str | None) -> None:
        if refresh_token is None:
            return
        verified = self.jwt_service.verify_jwt(refresh_token, "refresh")
        await self.user_repo.revoke_refresh_token(verified.payload["jti"], datetime.now(config.TIMEZONE))

    async def ensure_refresh_token_active(self, refresh_token: JWTRefreshToken) -> None:
        if await self.user_repo.is_refresh_token_revoked(refresh_token.payload["jti"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has been revoked.")

    def _default_login_id(self, email: str) -> str:
        base = email.split("@", 1)[0]
        return base[:40] if len(base) >= 6 else email[:40]

    def _mask_login_id(self, login_id: str) -> str:
        if len(login_id) <= 1:
            return f"{login_id}***"
        if len(login_id) <= 4:
            return f"{login_id[0]}***"
        return f"{login_id[:2]}{'*' * max(len(login_id) - 4, 3)}{login_id[-2:]}"

    def _digest(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    async def _record_failed_login(self, user: User, now: datetime) -> None:
        failed_count = int(user.failed_login_count or 0) + 1
        locked_until = None
        if failed_count >= config.LOGIN_FAILURE_LIMIT:
            locked_until = now + timedelta(minutes=config.LOGIN_SOFT_LOCK_MINUTES)
        await self.user_repo.record_login_failure(user, failed_count, locked_until)

    def _phone_verification_key(self, phone_number: str) -> str:
        return f"phone:{phone_number}"

    def _to_e164_kr(self, phone_number: str) -> str:
        normalized = normalize_phone_number(phone_number)
        if normalized.startswith("0"):
            return f"+82{normalized[1:]}"
        if normalized.startswith("82"):
            return f"+{normalized}"
        return f"+{normalized}"

    def _require_twilio_settings(self) -> tuple[str, str, str]:
        if not config.TWILIO_ACCOUNT_SID or not config.TWILIO_AUTH_TOKEN or not config.TWILIO_VERIFY_SERVICE_SID:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="휴대폰 인증 설정이 필요합니다."
            )
        return config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN, config.TWILIO_VERIFY_SERVICE_SID

    async def _send_twilio_verification(self, phone_number: str) -> None:
        sid, token, service_sid = self._require_twilio_settings()
        await self._call_twilio_verify(
            sid,
            token,
            f"https://verify.twilio.com/v2/Services/{service_sid}/Verifications",
            {"To": self._to_e164_kr(phone_number), "Channel": "sms"},
        )

    async def _check_twilio_verification(self, phone_number: str, code: str) -> bool:
        sid, token, service_sid = self._require_twilio_settings()
        response = await self._call_twilio_verify(
            sid,
            token,
            f"https://verify.twilio.com/v2/Services/{service_sid}/VerificationCheck",
            {"To": self._to_e164_kr(phone_number), "Code": code},
        )
        try:
            return json.loads(response.decode("utf-8")).get("status") == "approved"
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="휴대폰 인증 응답을 확인할 수 없습니다."
            ) from exc

    async def _call_twilio_verify(self, sid: str, token: str, url: str, data: dict[str, str]) -> bytes:
        def _request() -> bytes:
            encoded = urllib.parse.urlencode(data).encode("utf-8")
            auth = b64encode(f"{sid}:{token}".encode()).decode("ascii")
            request = urllib.request.Request(
                url,
                data=encoded,
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.read()

        try:
            return await asyncio.to_thread(_request)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="휴대폰 인증 처리에 실패했습니다."
            ) from exc
