from datetime import date, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, EmailStr, Field, model_validator

from app.core.validators import validate_birthday, validate_password, validate_phone_number
from app.models.users import Gender


class SignUpRequest(BaseModel):
    login_id: Annotated[str | None, Field(None, min_length=6, max_length=40)] = None
    email: Annotated[
        EmailStr,
        Field(None, max_length=40),
    ]
    password: Annotated[str, Field(min_length=8), AfterValidator(validate_password)]
    name: Annotated[str, Field(max_length=20)]
    gender: Gender
    birth_date: Annotated[date, AfterValidator(validate_birthday)]
    phone_number: Annotated[str, AfterValidator(validate_phone_number)]
    nickname: Annotated[str | None, Field(None, max_length=30)] = None
    address: Annotated[str | None, Field(None, max_length=255)] = None
    profile_image_url: Annotated[str | None, Field(None, max_length=500)] = None
    sensitive_data_agreed: bool = False
    marketing_agreed: bool = False


class LoginRequest(BaseModel):
    email: EmailStr | None = None
    login_id: Annotated[str | None, Field(None, min_length=6, max_length=40)] = None
    password: Annotated[str, Field(min_length=8)]

    @model_validator(mode="after")
    def validate_identifier(self) -> "LoginRequest":
        if self.email is None and self.login_id is None:
            raise ValueError("email 또는 login_id 중 하나는 필요합니다.")
        return self


class LoginResponse(BaseModel):
    access_token: str


class TokenRefreshResponse(LoginResponse): ...


class AvailabilityResponse(BaseModel):
    available: bool


class EmailVerificationSendRequest(BaseModel):
    email: EmailStr


class EmailVerificationSendResponse(BaseModel):
    detail: str
    debug_code: str | None = None


class EmailVerificationVerifyRequest(BaseModel):
    email: EmailStr
    code: Annotated[str, Field(min_length=6, max_length=6)]


class EmailVerificationVerifyResponse(BaseModel):
    verified: bool


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetRequestResponse(BaseModel):
    detail: str
    debug_token: str | None = None


class PasswordResetConfirmRequest(BaseModel):
    token: str
    new_password: Annotated[str, Field(min_length=8), AfterValidator(validate_password)]


class PasswordChangeRequest(BaseModel):
    current_password: Annotated[str, Field(min_length=8)]
    new_password: Annotated[str, Field(min_length=8), AfterValidator(validate_password)]


class SimpleMessageResponse(BaseModel):
    detail: str


class FirebaseSyncRequest(BaseModel):
    id_token: str | None = None


class FirebaseUserResponse(BaseModel):
    id: int
    email: str
    name: str
    nickname: str | None = None
    role: str
    is_active: bool
    auth_provider: str
    has_firebase_uid: bool
    created_at: datetime
    updated_at: datetime
