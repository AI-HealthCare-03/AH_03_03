from datetime import date, datetime
from types import SimpleNamespace

import pytest

from app.models.users import Gender
from app.services import users as user_service_module
from app.services.auth import ACCOUNT_NOT_FOUND_MESSAGE, AuthService
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
        self.revoked_refresh_tokens_for: tuple[int, datetime] | None = None
        self.deleted_password_tokens_for: int | None = None
        self.deleted_verification_codes_for: str | None = None
        self.updated_data: dict[str, object] | None = None

    async def revoke_refresh_tokens_by_user(self, user_id: int, revoked_at: datetime) -> int:
        self.revoked_refresh_tokens_for = (user_id, revoked_at)
        return 1

    async def delete_password_reset_tokens_by_user(self, user_id: int) -> int:
        self.deleted_password_tokens_for = user_id
        return 1

    async def delete_verification_codes_by_email(self, email: str) -> int:
        self.deleted_verification_codes_for = email
        return 1

    async def update_instance_allow_none(self, user: FakeUser, data: dict[str, object]) -> None:
        self.updated_data = data
        for key, value in data.items():
            setattr(user, key, value)


class FakeAvailabilityRepository:
    def __init__(self, users: list[FakeUser]) -> None:
        self.users = users

    async def exists_by_email(self, email: str) -> bool:
        return any(user.email == email for user in self.users)

    async def exists_by_login_id(self, login_id: str) -> bool:
        return any(user.login_id == login_id for user in self.users)

    async def exists_by_phone_number(self, phone_number: str) -> bool:
        return any(user.phone_number == phone_number for user in self.users)

    async def get_user_by_email(self, email: str) -> FakeUser | None:
        return next((user for user in self.users if user.email == email), None)


@pytest.mark.asyncio
async def test_deactivate_user_anonymizes_identifiers_and_clears_auth_artifacts(monkeypatch):
    monkeypatch.setattr(user_service_module, "in_transaction", lambda: DummyTransaction())
    cleanup_calls: list[tuple[str, int]] = []

    async def fake_delete_sensitive_service_data(self: UserManageService, user_id: int) -> None:
        cleanup_calls.append(("sensitive", user_id))

    async def fake_disable_notification_delivery(self: UserManageService, user_id: int) -> None:
        cleanup_calls.append(("notifications", user_id))

    async def fake_detach_family_links(self: UserManageService, user_id: int) -> None:
        cleanup_calls.append(("family", user_id))

    async def fake_detach_llm_and_rag_logs(self: UserManageService, user_id: int) -> None:
        cleanup_calls.append(("logs", user_id))

    monkeypatch.setattr(UserManageService, "_delete_sensitive_service_data", fake_delete_sensitive_service_data)
    monkeypatch.setattr(UserManageService, "_disable_notification_delivery", fake_disable_notification_delivery)
    monkeypatch.setattr(UserManageService, "_detach_family_links", fake_detach_family_links)
    monkeypatch.setattr(UserManageService, "_detach_llm_and_rag_logs", fake_detach_llm_and_rag_logs)

    user = FakeUser(
        id=42,
        login_id="demo42",
        email="demo42@example.com",
        hashed_password="old-hash",
        name="홍길동",
        nickname="길동",
        phone_number="01012345678",
        gender=Gender.MALE,
        birthday=date(1990, 1, 1),
        address="서울",
        profile_image_url="https://example.com/profile.png",
        is_active=True,
        is_admin=True,
        role="ADMIN",
        failed_login_count=3,
        locked_until=datetime(2026, 1, 1),
        email_verified_at=datetime(2026, 1, 1),
        deactivated_at=None,
    )
    repository = FakeUserRepository()
    service = UserManageService()
    service.repo = repository

    deactivated = await service.deactivate_user(user)

    assert deactivated.is_active is False
    assert deactivated.email == "deleted-42@deleted.local"
    assert deactivated.login_id is None
    assert deactivated.phone_number is None
    assert deactivated.name == "탈퇴회원"
    assert deactivated.nickname is None
    assert deactivated.address is None
    assert deactivated.profile_image_url is None
    assert deactivated.is_admin is False
    assert deactivated.role == "USER"
    assert deactivated.email_verified_at is None
    assert deactivated.deactivated_at is not None
    assert deactivated.hashed_password != "old-hash"

    assert repository.revoked_refresh_tokens_for is not None
    assert repository.revoked_refresh_tokens_for[0] == 42
    assert repository.deleted_password_tokens_for == 42
    assert repository.deleted_verification_codes_for == "demo42@example.com"
    assert cleanup_calls == [
        ("sensitive", 42),
        ("notifications", 42),
        ("family", 42),
        ("logs", 42),
    ]

    auth_service = AuthService()
    auth_service.user_repo = FakeAvailabilityRepository([deactivated])
    assert await auth_service.is_email_available("demo42@example.com") is True
    assert await auth_service.is_login_id_available("demo42") is True
    assert await auth_service.is_phone_number_available("01012345678") is True

    with pytest.raises(Exception) as exc_info:
        await auth_service.authenticate(type("Login", (), {"email": "demo42@example.com", "login_id": None})())

    assert getattr(exc_info.value, "detail", None) == ACCOUNT_NOT_FOUND_MESSAGE


@pytest.mark.asyncio
async def test_multiple_deactivated_users_can_keep_null_phone_numbers(monkeypatch):
    monkeypatch.setattr(user_service_module, "in_transaction", lambda: DummyTransaction())

    async def noop_cleanup(self: UserManageService, user_id: int) -> None:
        return None

    monkeypatch.setattr(UserManageService, "_delete_sensitive_service_data", noop_cleanup)
    monkeypatch.setattr(UserManageService, "_disable_notification_delivery", noop_cleanup)
    monkeypatch.setattr(UserManageService, "_detach_family_links", noop_cleanup)
    monkeypatch.setattr(UserManageService, "_detach_llm_and_rag_logs", noop_cleanup)

    service = UserManageService()
    service.repo = FakeUserRepository()
    users = [
        FakeUser(id=101, email="user101@example.com", phone_number="01011112222"),
        FakeUser(id=102, email="user102@example.com", phone_number="01033334444"),
    ]

    deactivated_users = [await service.deactivate_user(user) for user in users]

    assert [user.phone_number for user in deactivated_users] == [None, None]
