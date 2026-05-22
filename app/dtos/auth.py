from datetime import date
from typing import Annotated

from pydantic import AfterValidator, BaseModel, EmailStr, Field, model_validator

from app.core.validators import validate_birthday, validate_password, validate_phone_number
from app.models.users import Gender


def validate_required_text(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("필수 입력값입니다.")
    return stripped


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
    address: Annotated[str, Field(min_length=1, max_length=255), AfterValidator(validate_required_text)]
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
    message: str | None = None


class FindLoginIdRequest(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=20)]
    email: EmailStr | None = None
    phone_number: str | None = None

    @model_validator(mode="after")
    def validate_contact(self) -> "FindLoginIdRequest":
        if self.email is None and not self.phone_number:
            raise ValueError("email 또는 phone_number 중 하나는 필요합니다.")
        return self


class FindLoginIdResponse(BaseModel):
    found: bool
    masked_login_id: str | None = None
    message: str


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


class PhoneVerificationSendRequest(BaseModel):
    phone_number: Annotated[str, AfterValidator(validate_phone_number)]


class PhoneVerificationSendResponse(BaseModel):
    detail: str
    debug_code: str | None = None


class PhoneVerificationVerifyRequest(BaseModel):
    phone_number: Annotated[str, AfterValidator(validate_phone_number)]
    code: Annotated[str, Field(min_length=4, max_length=10)]


class PhoneVerificationVerifyResponse(BaseModel):
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
