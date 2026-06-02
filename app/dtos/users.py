from datetime import date, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, EmailStr, Field

from app.core.validators import optional_after_validator, validate_birthday, validate_phone_number
from app.dtos.base import BaseSerializerModel
from app.models.users import Gender


def validate_optional_nickname(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if len(normalized) < 2:
        raise ValueError("닉네임은 2자 이상 입력해주세요.")
    return normalized


class UserUpdateRequest(BaseModel):
    name: Annotated[str | None, Field(None, min_length=2, max_length=20)]
    login_id: Annotated[str | None, Field(None, min_length=6, max_length=40)] = None
    nickname: Annotated[
        str | None,
        Field(None, min_length=2, max_length=20),
        AfterValidator(validate_optional_nickname),
    ] = None
    email: Annotated[
        EmailStr | None,
        Field(None, max_length=40),
    ]
    phone_number: Annotated[
        str | None,
        Field(None, description="Available Format: +8201011112222, 01011112222, 010-1111-2222"),
        optional_after_validator(validate_phone_number),
    ]
    birthday: Annotated[
        date | None,
        Field(None, description="Date Format: YYYY-MM-DD"),
        optional_after_validator(validate_birthday),
    ]
    gender: Annotated[
        Gender | None,
        Field(None, description="'MALE' or 'FEMALE'"),
    ]
    address: Annotated[str | None, Field(None, max_length=255)] = None
    profile_image_url: Annotated[str | None, Field(None, max_length=500)] = None


class UserInfoResponse(BaseSerializerModel):
    id: int
    login_id: str | None = None
    name: str
    nickname: str | None = None
    email: str
    phone_number: str | None = None
    birthday: date
    gender: Gender
    address: str | None = None
    profile_image_url: str | None = None
    role: str = "USER"
    is_active: bool
    email_verified_at: datetime | None = None
    deactivated_at: datetime | None = None
    created_at: datetime
