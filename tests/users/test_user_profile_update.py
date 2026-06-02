from datetime import date
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.dtos.users import UserUpdateRequest
from app.models.users import Gender
from app.services import users as user_service_module
from app.services.users import UserManageService


class DummyTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeUser(SimpleNamespace):
    async def refresh_from_db(self) -> None:
        return None


class FakeUserRepository:
    def __init__(self) -> None:
        self.updated_data: dict[str, object] | None = None

    async def update_instance(self, user: FakeUser, data: dict[str, object]) -> None:
        self.updated_data = data
        for key, value in data.items():
            setattr(user, key, value)


class FakeAuthService:
    async def check_email_exists(self, email: str) -> None:
        return None

    async def check_login_id_exists(self, login_id: str) -> None:
        return None

    async def check_phone_number_exists(self, phone_number: str) -> None:
        return None


@pytest.mark.asyncio
async def test_update_user_trims_and_saves_nickname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(user_service_module, "in_transaction", lambda: DummyTransaction())
    user = FakeUser(
        id=1,
        login_id="profile01",
        name="사용자",
        nickname="이전닉",
        email="profile@example.com",
        phone_number=None,
        birthday=date(1990, 1, 1),
        gender=Gender.MALE,
    )
    repository = FakeUserRepository()
    service = UserManageService()
    service.repo = repository
    service.auth_service = FakeAuthService()

    updated = await service.update_user(user, UserUpdateRequest(nickname="  새닉네임  "))

    assert updated.nickname == "새닉네임"
    assert repository.updated_data == {"nickname": "새닉네임"}


@pytest.mark.parametrize("nickname", ["   ", "가", "가" * 21])
def test_user_update_rejects_invalid_nickname(nickname: str) -> None:
    with pytest.raises(ValidationError):
        UserUpdateRequest(nickname=nickname)
