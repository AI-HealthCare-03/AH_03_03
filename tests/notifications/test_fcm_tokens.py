from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.apis.v1.dependencies import get_request_user
from app.dtos.notifications import FCMTokenDeleteRequest, FCMTokenRegisterRequest
from app.main import app
from app.models.notifications import FCMTokenPlatform
from app.services import notifications as notification_service
from app.services import users as user_service_module
from app.services.users import UserManageService


class FakeNotificationRepository:
    def __init__(self) -> None:
        self.tokens: dict[str, SimpleNamespace] = {}
        self.deactivate_calls: list[tuple[int, str, datetime]] = []
        self.next_id = 1

    async def upsert_fcm_token(self, user_id: int, data: dict[str, object]) -> SimpleNamespace:
        token = str(data["token"])
        existing = self.tokens.get(token)
        if existing is None:
            existing = SimpleNamespace(id=self.next_id, created_at=datetime(2026, 5, 30, 9, 0, 0))
            self.next_id += 1
            self.tokens[token] = existing

        existing.user_id = user_id
        for key, value in data.items():
            setattr(existing, key, value)
        existing.updated_at = datetime(2026, 5, 30, 10, 0, 0)
        return existing

    async def deactivate_fcm_token(self, user_id: int, token: str, revoked_at: datetime) -> int:
        self.deactivate_calls.append((user_id, token, revoked_at))
        existing = self.tokens.get(token)
        if existing is None or existing.user_id != user_id:
            return 0
        existing.is_active = False
        existing.revoked_at = revoked_at
        return 1


@pytest.mark.asyncio
async def test_register_fcm_token_creates_and_reactivates_existing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = FakeNotificationRepository()
    monkeypatch.setattr(notification_service, "notification_repository", repository)
    request = FCMTokenRegisterRequest(
        token="fcm-token-create",
        platform=FCMTokenPlatform.WEB,
        device_id="browser-1",
    )

    created = await notification_service.register_fcm_token(
        1,
        request,
        request_user_agent="pytest-agent/1",
    )
    created.is_active = False
    created.revoked_at = datetime(2026, 1, 1)

    updated = await notification_service.register_fcm_token(
        1,
        request,
        request_user_agent="pytest-agent/2",
    )

    assert updated.id == created.id
    assert updated.user_id == 1
    assert updated.is_active is True
    assert updated.revoked_at is None
    assert updated.user_agent == "pytest-agent/2"
    assert len(repository.tokens) == 1


@pytest.mark.asyncio
async def test_register_fcm_token_reassigns_same_token_to_current_user(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = FakeNotificationRepository()
    monkeypatch.setattr(notification_service, "notification_repository", repository)
    await notification_service.register_fcm_token(
        10,
        FCMTokenRegisterRequest(token="fcm-token-reassign", platform=FCMTokenPlatform.IOS),
    )

    reassigned = await notification_service.register_fcm_token(
        20,
        FCMTokenRegisterRequest(token="fcm-token-reassign", platform=FCMTokenPlatform.ANDROID),
    )

    assert reassigned.user_id == 20
    assert reassigned.platform == FCMTokenPlatform.ANDROID
    assert len(repository.tokens) == 1


@pytest.mark.asyncio
async def test_deactivate_fcm_token_only_updates_current_user_token(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = FakeNotificationRepository()
    monkeypatch.setattr(notification_service, "notification_repository", repository)
    user_token = await notification_service.register_fcm_token(
        1,
        FCMTokenRegisterRequest(token="fcm-token-deactivate", platform=FCMTokenPlatform.WEB),
    )
    other_token = await notification_service.register_fcm_token(
        2,
        FCMTokenRegisterRequest(token="fcm-token-other", platform=FCMTokenPlatform.WEB),
    )

    not_mine_count = await notification_service.deactivate_fcm_token(
        1,
        FCMTokenDeleteRequest(token=other_token.token),
    )
    deactivated_count = await notification_service.deactivate_fcm_token(
        1,
        FCMTokenDeleteRequest(token=user_token.token),
    )

    assert not_mine_count == 0
    assert deactivated_count == 1
    assert user_token.is_active is False
    assert user_token.revoked_at is not None
    assert other_token.is_active is True
    assert other_token.revoked_at is None


@pytest.mark.asyncio
async def test_user_deactivation_disables_active_fcm_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeQuery:
        def __init__(self, name: str) -> None:
            self.name = name

        async def update(self, **kwargs) -> int:
            calls[self.name] = kwargs
            return 1

        async def delete(self) -> int:
            calls[self.name] = "deleted"
            return 1

    def fake_filter(name: str):
        return staticmethod(lambda **kwargs: FakeQuery(name))

    monkeypatch.setattr(user_service_module.UserFCMToken, "filter", fake_filter("fcm"))
    monkeypatch.setattr(user_service_module.ReminderSchedule, "filter", fake_filter("schedule"))
    monkeypatch.setattr(user_service_module.Notification, "filter", fake_filter("notification"))
    monkeypatch.setattr(user_service_module.NotificationLog, "filter", fake_filter("log"))

    await UserManageService()._disable_notification_delivery(1)

    assert calls["fcm"]["is_active"] is False
    assert calls["fcm"]["revoked_at"] is not None


def test_register_fcm_token_api_uses_current_user_and_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_current_user() -> SimpleNamespace:
        return SimpleNamespace(id=77)

    async def fake_register_fcm_token(
        user_id: int,
        request: FCMTokenRegisterRequest,
        *,
        request_user_agent: str | None = None,
    ) -> SimpleNamespace:
        captured["user_id"] = user_id
        captured["request"] = request
        captured["request_user_agent"] = request_user_agent
        now = datetime(2026, 5, 30, 10, 0, 0)
        return SimpleNamespace(
            id=1,
            user_id=user_id,
            token=request.token,
            platform=request.platform,
            device_id=request.device_id,
            user_agent=request_user_agent,
            is_active=True,
            last_seen_at=now,
            revoked_at=None,
            created_at=now,
            updated_at=now,
        )

    app.dependency_overrides[get_request_user] = fake_current_user
    monkeypatch.setattr(notification_service, "register_fcm_token", fake_register_fcm_token)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/notifications/fcm-tokens",
                json={"token": "api-fcm-token", "platform": "web", "device_id": "browser"},
                headers={"user-agent": "pytest-browser"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["token"] == "api-fcm-token"
    assert captured["user_id"] == 77
    assert captured["request_user_agent"] == "pytest-browser"


def test_deactivate_fcm_token_api_uses_current_user(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_current_user() -> SimpleNamespace:
        return SimpleNamespace(id=88)

    async def fake_deactivate_fcm_token(user_id: int, request: FCMTokenDeleteRequest) -> int:
        captured["user_id"] = user_id
        captured["token"] = request.token
        return 1

    app.dependency_overrides[get_request_user] = fake_current_user
    monkeypatch.setattr(notification_service, "deactivate_fcm_token", fake_deactivate_fcm_token)
    try:
        with TestClient(app) as client:
            response = client.request(
                "DELETE",
                "/api/v1/notifications/fcm-tokens",
                json={"token": "api-fcm-token"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"deactivated_count": 1}
    assert captured == {"user_id": 88, "token": "api-fcm-token"}
