import hashlib
import secrets
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
from app.dtos.auth import LoginRequest, PasswordChangeRequest, PasswordResetConfirmRequest, SignUpRequest
from app.models.users import User
from app.repositories.user_repository import UserRepository
from app.services.jwt import JwtService

VERIFICATION_CODE_TTL_MINUTES = 10
PASSWORD_RESET_TOKEN_TTL_MINUTES = 30


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

            return user

    async def authenticate(self, data: LoginRequest) -> User:
        if data.login_id is not None:
            user = await self.user_repo.get_user_by_login_id(data.login_id)
        else:
            user = await self.user_repo.get_user_by_email(str(data.email))
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="이메일 또는 비밀번호가 올바르지 않습니다."
            )

        # 비밀번호 검증
        if not verify_password(data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="이메일 또는 비밀번호가 올바르지 않습니다."
            )

        # 활성 사용자 체크
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_423_LOCKED, detail="비활성화된 계정입니다.")

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

    async def send_email_verification_code(self, email: str | EmailStr) -> str:
        code = f"{randint(0, 999999):06d}"
        expires_at = datetime.now(config.TIMEZONE) + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)
        await self.user_repo.create_verification_code(
            email=str(email),
            code_hash=self._digest(code),
            expires_at=expires_at,
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

    def _digest(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
