from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest

from app.dtos.notifications import NotificationCreateRequest
from app.models.notifications import NotificationChannel, NotificationLogStatus
from app.services import notification_email as notification_email_service
from app.services import notifications as notification_service
from app.services import service_jobs as service_job_service
from app.services.email_service import EmailDeliveryError


class FakeNotificationRepository:
    def __init__(self) -> None:
        self.notifications: list[SimpleNamespace] = []
        self.logs: list[dict[str, Any]] = []
        self.updated_logs: list[tuple[int, dict[str, Any]]] = []

    async def create_notification(self, user_id: int, data: dict[str, Any]) -> SimpleNamespace:
        notification = SimpleNamespace(id=len(self.notifications) + 1, user_id=user_id, **data)
        self.notifications.append(notification)
        return notification

    async def create_notification_log(self, user_id: int, data: dict[str, Any]) -> SimpleNamespace:
        log = {"id": len(self.logs) + 1, "user_id": user_id, **data}
        self.logs.append(log)
        return SimpleNamespace(**log)

    async def update_notification_log(self, notification_log_id: int, data: dict[str, Any]) -> SimpleNamespace | None:
        self.updated_logs.append((notification_log_id, data))
        for log in self.logs:
            if log["id"] == notification_log_id:
                log.update(data)
                return SimpleNamespace(**log)
        return None


@pytest.mark.asyncio
async def test_create_notification_without_send_email_does_not_enqueue_email(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = FakeNotificationRepository()
    enqueue_called = False

    async def fake_enqueue_notification_email_send(**kwargs):
        nonlocal enqueue_called
        enqueue_called = True

    monkeypatch.setattr(notification_service, "notification_repository", repository)
    monkeypatch.setattr(service_job_service, "enqueue_notification_email_send", fake_enqueue_notification_email_send)

    notification = await notification_service.create_notification(
        7,
        NotificationCreateRequest(
            notification_type="SYSTEM",
            title="서비스 알림",
            message="확인용 알림입니다.",
            send_email=False,
        ),
    )

    assert notification.id == 1
    assert repository.logs == []
    assert enqueue_called is False


@pytest.mark.asyncio
async def test_create_notification_with_send_email_enqueues_email_job(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = FakeNotificationRepository()
    enqueued: dict[str, Any] = {}

    async def fake_enqueue_notification_email_send(**kwargs):
        enqueued.update(kwargs)

    monkeypatch.setattr(notification_service, "notification_repository", repository)
    monkeypatch.setattr(service_job_service, "enqueue_notification_email_send", fake_enqueue_notification_email_send)

    notification = await notification_service.create_notification(
        7,
        NotificationCreateRequest(
            notification_type="SYSTEM",
            title="서비스 알림",
            message="확인용 알림입니다.",
            send_email=True,
            action_url="http://localhost:8080/notifications",
        ),
    )

    assert notification.id == 1
    assert repository.logs[0]["status"] == NotificationLogStatus.PENDING
    assert repository.logs[0]["channel"] == NotificationChannel.EMAIL
    assert enqueued == {
        "user_id": 7,
        "notification_id": 1,
        "notification_log_id": 1,
        "title": "서비스 알림",
        "message": "확인용 알림입니다.",
        "action_url": "http://localhost:8080/notifications",
    }


@pytest.mark.asyncio
async def test_create_notification_keeps_notification_when_email_enqueue_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = FakeNotificationRepository()

    async def fake_enqueue_notification_email_send(**kwargs):
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(notification_service, "notification_repository", repository)
    monkeypatch.setattr(service_job_service, "enqueue_notification_email_send", fake_enqueue_notification_email_send)

    notification = await notification_service.create_notification(
        7,
        NotificationCreateRequest(
            notification_type="SYSTEM",
            title="서비스 알림",
            message="확인용 알림입니다.",
            send_email=True,
        ),
    )

    assert notification.id == 1
    assert repository.logs[0]["status"] == NotificationLogStatus.FAILED
    assert repository.logs[0]["error_code"] == "notification_email_enqueue_failed"


@pytest.mark.asyncio
async def test_notification_email_delivery_skips_when_recipient_email_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    email_called = False

    class FakeEmailService:
        async def send_notification_email(self, *args, **kwargs):
            nonlocal email_called
            email_called = True
            return True

    async def fake_get_or_none(**kwargs):
        return SimpleNamespace(id=kwargs["id"], email="")

    monkeypatch.setattr(notification_email_service.User, "get_or_none", fake_get_or_none)

    result = await notification_email_service.deliver_notification_email_to_user(
        user_id=7,
        title="서비스 알림",
        message="확인용 알림입니다.",
        email_service=FakeEmailService(),
    )

    assert result.status == NotificationLogStatus.SKIPPED
    assert result.error_code == "recipient_email_missing"
    assert email_called is False


@pytest.mark.asyncio
async def test_notification_email_job_updates_failed_log_when_email_service_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updated_results: list[Any] = []
    marked_success: list[tuple[int, dict[str, Any]]] = []

    async def fake_job_payload(job_id: int) -> dict[str, Any]:
        assert job_id == 42
        return {
            "user_id": 7,
            "notification_log_id": 3,
            "title": "서비스 알림",
            "message": "확인용 알림입니다.",
        }

    async def fake_mark_processing(job_id: int) -> None:
        assert job_id == 42

    async def fake_mark_success(job_id: int, result: dict[str, Any]) -> None:
        marked_success.append((job_id, result))

    async def fake_deliver_notification_email_to_user(**kwargs):
        assert kwargs["user_id"] == 7
        return notification_email_service.NotificationEmailDeliveryResult(
            status=NotificationLogStatus.FAILED,
            sent=False,
            provider="smtp",
            error_code="email_delivery_failed",
            error_message="email_delivery_failed",
        )

    async def fake_update_notification_log_with_email_result(notification_log_id: int, delivery_result):
        updated_results.append((notification_log_id, delivery_result))

    monkeypatch.setattr(service_job_service, "_job_payload", fake_job_payload)
    monkeypatch.setattr(service_job_service.async_job_service, "mark_processing", fake_mark_processing)
    monkeypatch.setattr(service_job_service.async_job_service, "mark_success", fake_mark_success)
    monkeypatch.setattr(
        notification_email_service,
        "deliver_notification_email_to_user",
        fake_deliver_notification_email_to_user,
    )
    monkeypatch.setattr(
        notification_service,
        "update_notification_log_with_email_result",
        fake_update_notification_log_with_email_result,
    )

    result = await service_job_service.handle_notification_email_send(42)

    assert result == {"sent": False, "status": "FAILED", "kind": "notification_email"}
    assert updated_results[0][0] == 3
    assert updated_results[0][1].status == NotificationLogStatus.FAILED
    assert marked_success[0][1]["status"] == "FAILED"


@pytest.mark.asyncio
async def test_notification_email_delivery_returns_failed_when_email_service_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEmailService:
        async def send_notification_email(self, *args, **kwargs):
            raise EmailDeliveryError("smtp failed")

    async def fake_get_or_none(**kwargs):
        return SimpleNamespace(id=kwargs["id"], email="user@example.com")

    monkeypatch.setattr(notification_email_service.User, "get_or_none", fake_get_or_none)

    result = await notification_email_service.deliver_notification_email_to_user(
        user_id=7,
        title="서비스 알림",
        message="확인용 알림입니다.",
        email_service=FakeEmailService(),
    )

    assert result.status == NotificationLogStatus.FAILED
    assert result.error_code == "email_delivery_failed"
    assert result.sent is False


@pytest.mark.asyncio
async def test_smoke_notification_email_script_dry_run_does_not_send(capsys: pytest.CaptureFixture[str]) -> None:
    module_path = "scripts/qa/smoke_notification_email.py"
    spec = importlib.util.spec_from_file_location("smoke_notification_email", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    exit_code = await module._run(
        SimpleNamespace(
            confirm_send=False,
            to_email=None,
            user_id=None,
            title="알림 이메일 스모크",
            message="확인용 메시지입니다.",
            action_url=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DRY_RUN_ONLY=true" in captured.out
    assert "No email was sent" in captured.out
