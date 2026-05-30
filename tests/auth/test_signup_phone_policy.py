from datetime import date
from types import SimpleNamespace

import pytest

from app.dtos.auth import SignUpRequest
from app.services import auth as auth_service_module
from app.services.auth import AuthService


class DummyTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeSignupRepository:
    def __init__(self) -> None:
        self.created_user_payload: dict[str, object] | None = None

    async def exists_by_login_id(self, login_id: str) -> bool:
        return False

    async def exists_by_email(self, email: str) -> bool:
        return False

    async def exists_by_phone_number(self, phone_number: str) -> bool:
        return False

    async def create_user(self, **kwargs):
        self.created_user_payload = kwargs
        return SimpleNamespace(id=1, **kwargs)

    async def create_user_consent(self, user_id: int, sensitive_data_agreed: bool, marketing_agreed: bool):
        return SimpleNamespace(
            user_id=user_id,
            sensitive_data_agreed=sensitive_data_agreed,
            marketing_agreed=marketing_agreed,
        )


@pytest.fixture
def signup_service(monkeypatch):
    monkeypatch.setattr(auth_service_module, "in_transaction", lambda: DummyTransaction())

    async def fake_ensure_email_verified(self: AuthService, email: str) -> None:
        return None

    async def fake_get_user_setting_by_user(user_id: int):
        return SimpleNamespace(id=1, user_id=user_id)

    monkeypatch.setattr(AuthService, "ensure_email_verified", fake_ensure_email_verified)
    monkeypatch.setattr(
        auth_service_module.setting_repository, "get_user_setting_by_user", fake_get_user_setting_by_user
    )

    repository = FakeSignupRepository()
    service = AuthService()
    service.user_repo = repository
    return service, repository


@pytest.mark.asyncio
async def test_signup_without_phone_number_stores_none(signup_service):
    service, repository = signup_service

    await service.signup(
        SignUpRequest(
            login_id="signupnone",
            email="signup-none@example.com",
            password="Password123!",
            name="테스터",
            gender="MALE",
            birth_date=date(1990, 1, 1),
        )
    )

    assert repository.created_user_payload is not None
    assert repository.created_user_payload["phone_number"] is None


@pytest.mark.asyncio
async def test_signup_with_phone_number_stores_normalized_value(signup_service):
    service, repository = signup_service

    await service.signup(
        SignUpRequest(
            login_id="signupphone",
            email="signup-phone@example.com",
            password="Password123!",
            name="테스터",
            gender="MALE",
            birth_date=date(1990, 1, 1),
            phone_number="+82 10-1234-5678",
        )
    )

    assert repository.created_user_payload is not None
    assert repository.created_user_payload["phone_number"] == "01012345678"
