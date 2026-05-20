from datetime import date, datetime
from typing import Any

from pydantic import EmailStr

from app.core import config
from app.models.users import Gender, PasswordResetToken, RefreshToken, User, UserConsent, VerificationCode

ALLOWED_UPDATE_FIELDS = [
    "login_id",
    "name",
    "nickname",
    "email",
    "phone_number",
    "gender",
    "birthday",
    "address",
    "profile_image_url",
]
UPDATED_AT_FIELD = "updated_at"


class UserRepository:
    def __init__(self):
        self._model = User

    async def get_all(self):
        return await self._model.all()

    async def get_user(self, user_id: int) -> User | None:
        return await self._model.get_or_none(id=user_id)

    async def create_user(
        self,
        email: str | EmailStr,
        hashed_password: str,
        name: str,
        phone_number: str,
        gender: Gender,
        birthday: date,
        *,
        login_id: str | None = None,
        firebase_uid: str | None = None,
        auth_provider: str = "local",
        nickname: str | None = None,
        address: str | None = None,
        profile_image_url: str | None = None,
        email_verified_at: datetime | None = None,
        is_active: bool = True,
        is_admin: bool = False,
        role: str = "USER",
    ) -> User:
        return await self._model.create(
            login_id=login_id,
            firebase_uid=firebase_uid,
            auth_provider=auth_provider,
            email=email,
            hashed_password=hashed_password,
            name=name,
            nickname=nickname,
            phone_number=phone_number,
            gender=gender,
            birthday=birthday,
            address=address,
            profile_image_url=profile_image_url,
            is_active=is_active,
            is_admin=is_admin,
            role=role,
            email_verified_at=email_verified_at,
        )

    async def get_user_by_email(self, email: str) -> User | None:
        return await self._model.get_or_none(email=email)

    async def get_user_by_login_id(self, login_id: str) -> User | None:
        return await self._model.get_or_none(login_id=login_id)

    async def get_user_by_firebase_uid(self, firebase_uid: str) -> User | None:
        return await self._model.get_or_none(firebase_uid=firebase_uid)

    async def exists_by_email(self, email: str) -> bool:
        return await self._model.filter(email=email).exists()

    async def exists_by_login_id(self, login_id: str) -> bool:
        return await self._model.filter(login_id=login_id).exists()

    async def exists_by_phone_number(self, phone_number: str) -> bool:
        return await self._model.filter(phone_number=phone_number).exists()

    async def update_last_login(self, user_id: int) -> None:
        await self._model.filter(id=user_id).update(last_login=datetime.now(config.TIMEZONE))

    async def update_instance(self, user: User, data: dict[str, Any]) -> None:
        update_fields = []
        for key, value in data.items():
            if value is not None:
                setattr(user, key, value)
                update_fields.append(key)
        if update_fields:
            user.updated_at = datetime.now(config.TIMEZONE)
            update_fields.append(UPDATED_AT_FIELD)
            await user.save(update_fields=update_fields)

    async def update_user_fields(self, user_id: int, data: dict[str, Any]) -> User | None:
        user = await self.get_user(user_id)
        if user is None:
            return None
        await self.update_instance(user, data)
        await user.refresh_from_db()
        return user

    async def create_verification_code(
        self,
        email: str,
        code_hash: str,
        expires_at: datetime,
        purpose: str = "EMAIL_VERIFICATION",
    ) -> VerificationCode:
        return await VerificationCode.create(
            email=email,
            code_hash=code_hash,
            expires_at=expires_at,
            purpose=purpose,
        )

    async def get_latest_verification_code(
        self, email: str, purpose: str = "EMAIL_VERIFICATION"
    ) -> VerificationCode | None:
        return (
            await VerificationCode.filter(email=email, purpose=purpose, is_used=False).order_by("-created_at").first()
        )

    async def mark_verification_code_used(self, code: VerificationCode, verified_at: datetime) -> VerificationCode:
        code.is_used = True
        code.verified_at = verified_at
        await code.save(update_fields=["is_used", "verified_at"])
        return code

    async def create_password_reset_token(
        self,
        user_id: int,
        token_hash: str,
        expires_at: datetime,
    ) -> PasswordResetToken:
        return await PasswordResetToken.create(user_id=user_id, token_hash=token_hash, expires_at=expires_at)

    async def get_password_reset_token(self, token_hash: str) -> PasswordResetToken | None:
        return await PasswordResetToken.get_or_none(token_hash=token_hash, is_used=False)

    async def mark_password_reset_token_used(self, token: PasswordResetToken, used_at: datetime) -> PasswordResetToken:
        token.is_used = True
        token.used_at = used_at
        await token.save(update_fields=["is_used", "used_at"])
        return token

    async def update_password(self, user_id: int, hashed_password: str) -> None:
        await self._model.filter(id=user_id).update(
            hashed_password=hashed_password, updated_at=datetime.now(config.TIMEZONE)
        )

    async def create_refresh_token(self, user_id: int, token_jti: str, expires_at: datetime) -> RefreshToken:
        return await RefreshToken.create(user_id=user_id, token_jti=token_jti, expires_at=expires_at)

    async def revoke_refresh_token(self, token_jti: str, revoked_at: datetime) -> int:
        return await RefreshToken.filter(token_jti=token_jti, revoked_at=None).update(revoked_at=revoked_at)

    async def is_refresh_token_revoked(self, token_jti: str) -> bool:
        token = await RefreshToken.get_or_none(token_jti=token_jti)
        return token is not None and token.revoked_at is not None

    async def create_user_consent(
        self,
        user_id: int,
        sensitive_data_agreed: bool = False,
        marketing_agreed: bool = False,
    ) -> UserConsent:
        return await UserConsent.create(
            user_id=user_id,
            sensitive_data_agreed=sensitive_data_agreed,
            marketing_agreed=marketing_agreed,
        )

    async def sync_firebase_user(
        self,
        *,
        firebase_uid: str,
        email: str,
        name: str,
        nickname: str | None = None,
        profile_image_url: str | None = None,
        email_verified_at: datetime | None = None,
    ) -> User:
        user = await self.get_user_by_firebase_uid(firebase_uid)
        if user is None:
            user = await self.get_user_by_email(email)

        data = {
            "firebase_uid": firebase_uid,
            "auth_provider": "firebase",
            "email": email,
            "name": name,
            "nickname": nickname,
            "profile_image_url": profile_image_url,
            "email_verified_at": email_verified_at,
            "role": "USER",
        }

        if user is not None:
            await self.update_instance(user, {key: value for key, value in data.items() if value is not None})
            await user.refresh_from_db()
            return user

        return await self.create_user(
            login_id=self._build_firebase_login_id(firebase_uid),
            firebase_uid=firebase_uid,
            auth_provider="firebase",
            email=email,
            hashed_password="firebase-auth-user",
            name=name,
            nickname=nickname,
            phone_number=self._build_firebase_phone_number(firebase_uid),
            gender=Gender.MALE,
            birthday=date(1970, 1, 1),
            profile_image_url=profile_image_url,
            email_verified_at=email_verified_at,
            role="USER",
        )

    def _build_firebase_login_id(self, firebase_uid: str) -> str:
        return f"fb_{firebase_uid[:37]}"

    def _build_firebase_phone_number(self, firebase_uid: str) -> str:
        digits = "".join(str(ord(char) % 10) for char in firebase_uid)
        return f"fb{digits[:9]}".ljust(11, "0")
