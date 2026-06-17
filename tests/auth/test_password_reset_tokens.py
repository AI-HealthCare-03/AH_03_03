from __future__ import annotations

import urllib.parse
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.core import config
from app.dtos.auth import PasswordResetConfirmRequest
from app.services import auth as auth_service_module
from app.services import service_jobs
from app.services.auth import AuthService


class FakePasswordResetRepository:
    def __init__(self) -> None:
        self.user = SimpleNamespace(id=7, email="reset@example.com")
        self.tokens: list[SimpleNamespace] = []
        self.updated_passwords: list[tuple[int, str]] = []

    async def get_user_by_email(self, email: str) -> SimpleNamespace | None:
        return self.user if email == self.user.email else None

    async def mark_active_password_reset_tokens_used(self, user_id: int, used_at: datetime) -> int:
        updated = 0
        for token in self.tokens:
            if token.user_id == user_id and not token.is_used:
                token.is_used = True
                token.used_at = used_at
                updated += 1
        return updated

    async def create_password_reset_token(
        self,
        user_id: int,
        token_hash: str,
        expires_at: datetime,
    ) -> SimpleNamespace:
        token = SimpleNamespace(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            is_used=False,
            used_at=None,
        )
        self.tokens.append(token)
        return token

    async def get_password_reset_token(self, token_hash: str) -> SimpleNamespace | None:
        for token in self.tokens:
            if token.token_hash == token_hash and not token.is_used:
                return token
        return None

    async def update_password(self, user_id: int, hashed_password: str) -> None:
        self.updated_passwords.append((user_id, hashed_password))

    async def mark_password_reset_token_used(
        self,
        token: SimpleNamespace,
        used_at: datetime,
    ) -> SimpleNamespace:
        token.is_used = True
        token.used_at = used_at
        return token


def _token_from_reset_url(reset_url: str) -> str:
    parsed = urllib.parse.urlparse(reset_url)
    values = urllib.parse.parse_qs(parsed.query).get("token")
    assert values
    return values[0]


@pytest.mark.asyncio
async def test_password_reset_reissue_invalidates_previous_token(monkeypatch) -> None:
    repository = FakePasswordResetRepository()
    sent_reset_urls: list[str] = []
    token_values = iter(["first-reset-token", "second-reset-token"])

    async def fake_enqueue_password_reset_email_send(*, email: str, reset_url: str) -> None:
        assert email == repository.user.email
        sent_reset_urls.append(reset_url)

    monkeypatch.setattr(AuthService, "_ensure_email_delivery_available", lambda self: None)
    monkeypatch.setattr(auth_service_module.secrets, "token_urlsafe", lambda size: next(token_values))
    monkeypatch.setattr(service_jobs, "enqueue_password_reset_email_send", fake_enqueue_password_reset_email_send)
    monkeypatch.setattr(auth_service_module.config, "FRONTEND_BASE_URL", "https://healthladder.example")

    service = AuthService()
    service.user_repo = repository

    first_token = await service.request_password_reset(repository.user.email)
    second_token = await service.request_password_reset(repository.user.email)

    assert first_token != second_token
    assert len(sent_reset_urls) == 2
    assert _token_from_reset_url(sent_reset_urls[0]) != _token_from_reset_url(sent_reset_urls[1])

    with pytest.raises(HTTPException):
        await service.confirm_password_reset(
            PasswordResetConfirmRequest(token=first_token, new_password="NewPassword123!")
        )

    await service.confirm_password_reset(PasswordResetConfirmRequest(token=second_token, new_password="NewPassword123!"))

    assert repository.updated_passwords

    with pytest.raises(HTTPException):
        await service.confirm_password_reset(
            PasswordResetConfirmRequest(token=second_token, new_password="NewPassword123!")
        )


@pytest.mark.asyncio
async def test_expired_password_reset_token_fails(monkeypatch) -> None:
    repository = FakePasswordResetRepository()
    expired_token = "expired-reset-token"
    service = AuthService()
    service.user_repo = repository

    await repository.create_password_reset_token(
        user_id=repository.user.id,
        token_hash=service._digest(expired_token),
        expires_at=datetime.now(config.TIMEZONE) - timedelta(minutes=1),
    )

    with pytest.raises(HTTPException):
        await service.confirm_password_reset(
            PasswordResetConfirmRequest(token=expired_token, new_password="NewPassword123!")
        )
