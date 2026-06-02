from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.core.utils.security import hash_password
from app.dtos.auth import LoginRequest
from app.services.auth import ACCOUNT_NOT_FOUND_MESSAGE, INVALID_LOGIN_MESSAGE, AuthService


class FakeUserRepository:
    def __init__(self, user: SimpleNamespace | None = None):
        self.user = user
        self.failed_login_recorded = False

    async def get_user_by_email(self, email: str) -> SimpleNamespace | None:
        return self.user if self.user and self.user.email == email else None

    async def get_user_by_login_id(self, login_id: str) -> SimpleNamespace | None:
        return self.user if self.user and self.user.login_id == login_id else None

    async def record_login_failure(self, user: SimpleNamespace, failed_count: int, locked_until: object) -> None:
        self.failed_login_recorded = True
        user.failed_login_count = failed_count
        user.locked_until = locked_until


@pytest.mark.asyncio
async def test_authenticate_reports_missing_account() -> None:
    service = AuthService()
    service.user_repo = FakeUserRepository()

    with pytest.raises(HTTPException) as exc_info:
        await service.authenticate(LoginRequest(email="missing@example.com", password="Password123!"))

    assert exc_info.value.detail == ACCOUNT_NOT_FOUND_MESSAGE


@pytest.mark.asyncio
async def test_authenticate_reports_wrong_password() -> None:
    user = SimpleNamespace(
        id=1,
        login_id="loginuser",
        email="login@example.com",
        hashed_password=hash_password("Password123!"),
        failed_login_count=0,
        locked_until=None,
        is_active=True,
    )
    repository = FakeUserRepository(user)
    service = AuthService()
    service.user_repo = repository

    with pytest.raises(HTTPException) as exc_info:
        await service.authenticate(LoginRequest(email="login@example.com", password="WrongPassword123!"))

    assert exc_info.value.detail == INVALID_LOGIN_MESSAGE
    assert repository.failed_login_recorded is True
