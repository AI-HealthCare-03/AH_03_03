from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.services import family as family_service


class AwaitableList:
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    def __await__(self):
        async def _result() -> list[Any]:
            return self.items

        return _result().__await__()


class NotificationQuery:
    def __init__(self, exists: bool = False) -> None:
        self._exists = exists

    async def exists(self) -> bool:
        return self._exists


def _context() -> family_service.FamilyChallengeAlertContext:
    return family_service.FamilyChallengeAlertContext(
        owner_user_id=1,
        owner_display_name="동욱",
        user_challenge_id=77,
    )


def _setting(
    *,
    notify_completed: bool = True,
    channel_in_app: bool = True,
    channel_push: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        notify_challenge_missed=False,
        notify_challenge_completed=notify_completed,
        notify_medication_missed=False,
        notify_diet_missed=False,
        notify_report_updated=False,
        channel_in_app=channel_in_app,
        channel_push=channel_push,
    )


def _patch_family_alert_base(
    monkeypatch: pytest.MonkeyPatch,
    *,
    share_settings: list[SimpleNamespace],
    notification_setting: SimpleNamespace | None,
    created_notifications: list[SimpleNamespace] | None = None,
    patch_push: bool = True,
) -> list[SimpleNamespace]:
    created = created_notifications if created_notifications is not None else []

    async def fake_context(user_challenge_id: int) -> family_service.FamilyChallengeAlertContext:
        assert user_challenge_id == 77
        return _context()

    async def fake_get_notification_setting(**kwargs: object) -> SimpleNamespace | None:
        assert kwargs == {"owner_user_id": 1, "family_user_id": 2}
        return notification_setting

    async def fake_create_notification(user_id: int, request: object) -> SimpleNamespace:
        notification = SimpleNamespace(id=len(created) + 1, user_id=user_id, request=request)
        created.append(notification)
        return notification

    async def fake_send_family_push_if_allowed(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(family_service, "_get_completed_challenge_alert_context", fake_context)
    monkeypatch.setattr(
        family_service.FamilyShareSetting,
        "filter",
        staticmethod(lambda **kwargs: AwaitableList(share_settings)),
    )
    monkeypatch.setattr(
        family_service.FamilyNotificationSetting,
        "get_or_none",
        staticmethod(fake_get_notification_setting),
    )
    monkeypatch.setattr(
        family_service.Notification,
        "filter",
        staticmethod(lambda **kwargs: NotificationQuery(False)),
    )
    monkeypatch.setattr(family_service.notification_service, "create_notification", fake_create_notification)
    if patch_push:
        monkeypatch.setattr(family_service, "_send_family_push_if_allowed", fake_send_family_push_if_allowed)
    return created


@pytest.mark.asyncio
async def test_family_challenge_alert_skips_when_owner_does_not_share(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_family_alert_base(
        monkeypatch,
        share_settings=[],
        notification_setting=_setting(),
    )

    result = await family_service.notify_family_challenge_completed(77)

    assert result == []
    assert created == []


@pytest.mark.asyncio
async def test_family_challenge_alert_skips_when_family_notification_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_family_alert_base(
        monkeypatch,
        share_settings=[SimpleNamespace(viewer_user_id=2)],
        notification_setting=_setting(notify_completed=False),
    )

    result = await family_service.notify_family_challenge_completed(77)

    assert result == []
    assert created == []


@pytest.mark.asyncio
async def test_family_challenge_alert_creates_internal_notification_without_sensitive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_family_alert_base(
        monkeypatch,
        share_settings=[SimpleNamespace(viewer_user_id=2)],
        notification_setting=_setting(),
    )

    result = await family_service.notify_family_challenge_completed(77)

    assert result == created
    assert len(created) == 1
    request = created[0].request
    assert request.title == "가족 챌린지 알림"
    assert request.message == "동욱님이 이번 챌린지 목표를 달성했어요."
    assert request.related_type == family_service.FAMILY_CHALLENGE_COMPLETED_RELATED_TYPE
    for sensitive_value in ("120", "혈압", "혈당", "체중", "위험도", "OCR"):
        assert sensitive_value not in request.message


@pytest.mark.asyncio
async def test_family_challenge_alert_sends_push_when_enabled_with_active_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[SimpleNamespace] = []
    push_calls: list[dict[str, object]] = []
    log_calls: list[dict[str, object]] = []
    _patch_family_alert_base(
        monkeypatch,
        share_settings=[SimpleNamespace(viewer_user_id=2)],
        notification_setting=_setting(channel_push=True),
        created_notifications=created,
        patch_push=False,
    )

    async def fake_list_active_fcm_tokens(user_id: int) -> list[str]:
        assert user_id == 2
        return ["family-token"]

    async def fake_record_push_result(**kwargs: object) -> None:
        log_calls.append(kwargs)

    class FakeFCMService:
        def send_to_tokens(self, **kwargs: object) -> SimpleNamespace:
            push_calls.append(kwargs)
            return SimpleNamespace(success_count=1, failure_count=0, provider_message_ids=["msg-1"])

    monkeypatch.setattr(family_service, "_list_active_fcm_tokens", fake_list_active_fcm_tokens)
    monkeypatch.setattr(family_service, "_record_push_result", fake_record_push_result)
    monkeypatch.setattr(family_service, "FCMService", FakeFCMService)

    result = await family_service.notify_family_challenge_completed(77)

    assert len(result) == 1
    assert push_calls == [
        {
            "tokens": ["family-token"],
            "title": "가족 챌린지 알림",
            "body": "동욱님이 이번 챌린지 목표를 달성했어요.",
            "data": {"type": family_service.FAMILY_CHALLENGE_COMPLETED_RELATED_TYPE, "user_challenge_id": "77"},
        }
    ]
    assert log_calls


@pytest.mark.asyncio
async def test_family_challenge_alert_keeps_internal_notification_when_push_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[SimpleNamespace] = []
    log_calls: list[dict[str, object]] = []
    _patch_family_alert_base(
        monkeypatch,
        share_settings=[SimpleNamespace(viewer_user_id=2)],
        notification_setting=_setting(channel_push=True),
        created_notifications=created,
        patch_push=False,
    )

    async def fake_list_active_fcm_tokens(user_id: int) -> list[str]:
        assert user_id == 2
        return ["family-token"]

    async def fake_record_push_result(**kwargs: object) -> None:
        log_calls.append(kwargs)

    class FailingFCMService:
        def send_to_tokens(self, **kwargs: object) -> SimpleNamespace:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(family_service, "_list_active_fcm_tokens", fake_list_active_fcm_tokens)
    monkeypatch.setattr(family_service, "_record_push_result", fake_record_push_result)
    monkeypatch.setattr(family_service, "FCMService", FailingFCMService)

    result = await family_service.notify_family_challenge_completed(77)

    assert result == created
    assert len(created) == 1
    assert log_calls
    assert log_calls[0]["error_code"] == "RuntimeError"


@pytest.mark.asyncio
async def test_family_challenge_alert_skips_canceled_or_not_finalized_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[SimpleNamespace] = []

    async def fake_context(user_challenge_id: int) -> None:
        assert user_challenge_id == 77
        return None

    async def fake_create_notification(user_id: int, request: object) -> SimpleNamespace:
        notification = SimpleNamespace(id=1, user_id=user_id, request=request)
        created.append(notification)
        return notification

    monkeypatch.setattr(family_service, "_get_completed_challenge_alert_context", fake_context)
    monkeypatch.setattr(family_service.notification_service, "create_notification", fake_create_notification)

    result = await family_service.notify_family_challenge_completed(77)

    assert result == []
    assert created == []
