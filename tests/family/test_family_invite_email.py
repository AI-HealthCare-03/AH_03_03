from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.apis.v1 import family_routers
from app.apis.v1.dependencies import get_request_user
from app.core.config import Env
from app.dtos.family import FamilyInviteCreateRequest
from app.main import app
from app.models.family import FamilyInviteStatus, FamilyMemberRole, FamilyRelationType
from app.services import family as family_service
from app.services.email_service import EmailService


@pytest.mark.asyncio
async def test_family_invite_creation_sends_email_with_invite_link(monkeypatch: pytest.MonkeyPatch) -> None:
    email_jobs: list[dict[str, str]] = []
    create_payloads: list[dict[str, object]] = []
    inviter = SimpleNamespace(id=1, nickname="동욱", name="홍동욱", login_id="dongwook")
    family = SimpleNamespace(id=10)

    async def fake_ensure_owner(user: object, family_id: int) -> SimpleNamespace:
        assert user is inviter
        assert family_id == 10
        return family

    async def fake_create_invite(**kwargs: object) -> SimpleNamespace:
        create_payloads.append(kwargs)
        return SimpleNamespace(
            id=3,
            family_id=10,
            inviter_user_id=1,
            invitee_user_id=None,
            invitee_email=kwargs["invitee_email"],
            invitee_phone=None,
            relation_type=kwargs["relation_type"],
            member_role=kwargs["member_role"],
            status=FamilyInviteStatus.PENDING,
            expires_at=kwargs["expires_at"],
            used_at=None,
            created_at=datetime(2026, 5, 30, 10, 0, 0),
        )

    async def fake_enqueue_family_invite_email_send(**kwargs: str) -> None:
        email_jobs.append(kwargs)

    monkeypatch.setattr(family_service, "_ensure_owner", fake_ensure_owner)
    monkeypatch.setattr(family_service.secrets, "token_urlsafe", lambda length: "invite-token")
    monkeypatch.setattr(family_service.FamilyInvite, "create", staticmethod(fake_create_invite))
    monkeypatch.setattr(
        family_service.service_jobs,
        "enqueue_family_invite_email_send",
        fake_enqueue_family_invite_email_send,
    )
    monkeypatch.setattr(family_service.config, "FRONTEND_BASE_URL", "http://localhost:8080")

    invite, invite_code = await family_service.create_family_invite(
        inviter,
        10,
        FamilyInviteCreateRequest(
            invitee_email="Family@Example.com",
            relation_type=FamilyRelationType.SPOUSE,
            member_role=FamilyMemberRole.MEMBER,
        ),
    )

    assert invite.id == 3
    assert invite_code == "invite-token"
    assert create_payloads[0]["invitee_email"] == "family@example.com"
    assert create_payloads[0]["code_hash"] == family_service._digest("invite-token")
    assert email_jobs == [
        {
            "recipient_email": "family@example.com",
            "inviter_display_name": "동욱",
            "invite_code": "invite-token",
            "invite_url": "http://localhost:8080/family/invitations/accept?code=invite-token",
            "expires_at_text": invite.expires_at.astimezone(family_service.config.TIMEZONE).strftime("%Y-%m-%d %H:%M"),
        }
    ]


@pytest.mark.asyncio
async def test_family_invite_email_send_is_safe_when_email_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.email_service.config.EMAIL_ENABLED", False)
    monkeypatch.setattr("app.services.email_service.config.ENV", Env.LOCAL)

    sent = await EmailService().send_family_invite_email(
        "family@example.com",
        inviter_display_name="동욱",
        invite_code="invite-token",
        invite_url="http://localhost:8080/family/invitations/accept?code=invite-token",
        expires_at_text="2026-05-31 10:00",
    )

    assert sent is False


@pytest.mark.asyncio
async def test_family_invite_email_body_does_not_include_sensitive_health_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    async def fake_send_email(
        self: EmailService,
        recipient: str,
        subject: str,
        body: str,
        html_body: str | None = None,
    ) -> bool:
        captured["recipient"] = recipient
        captured["subject"] = subject
        captured["body"] = body
        captured["html_body"] = html_body or ""
        return True

    monkeypatch.setattr(EmailService, "_send_email", fake_send_email)

    sent = await EmailService().send_family_invite_email(
        "family@example.com",
        inviter_display_name="동욱",
        invite_code="invite-token",
        invite_url="http://localhost:8080/family/invitations/accept?code=invite-token",
        expires_at_text="2026-05-31 10:00",
    )

    assert sent is True
    assert captured["subject"] == "[Health Ladder] 가족 건강관리 초대 안내"
    assert "가족 건강관리 기능에 초대되었습니다." in captured["body"]
    assert "초대 코드: invite-token" in captured["body"]
    assert "http://localhost:8080/family/invitations/accept?code=invite-token" in captured["body"]
    assert '<p style="font-size: 20px; font-weight: 700;">초대 코드: invite-token</p>' in captured["html_body"]
    assert '<a href="http://localhost:8080/family/invitations/accept?code=invite-token"' in captured["html_body"]
    for sensitive_value in ("120", "혈압", "혈당", "체중", "질병 위험도", "OCR"):
        assert sensitive_value not in captured["body"]
        assert sensitive_value not in captured["html_body"]


@pytest.mark.asyncio
async def test_family_invite_rejects_expired_code(monkeypatch: pytest.MonkeyPatch) -> None:
    invite = SimpleNamespace(
        id=9,
        status=FamilyInviteStatus.PENDING,
        used_at=None,
        expires_at=family_service._now() - timedelta(minutes=1),
    )
    updates: list[dict[str, object]] = []

    class FakeInviteQuery:
        async def update(self, **kwargs: object) -> int:
            updates.append(kwargs)
            return 1

    monkeypatch.setattr(family_service.FamilyInvite, "filter", staticmethod(lambda **kwargs: FakeInviteQuery()))

    with pytest.raises(HTTPException) as exc:
        await family_service._ensure_invite_usable(invite)

    assert exc.value.status_code == 400
    assert exc.value.detail == "만료된 가족 초대입니다."
    assert updates == [{"status": FamilyInviteStatus.EXPIRED}]


@pytest.mark.asyncio
async def test_family_invite_rejects_reused_code() -> None:
    invite = SimpleNamespace(
        status=FamilyInviteStatus.ACCEPTED,
        used_at=family_service._now(),
        expires_at=family_service._now() + timedelta(hours=1),
    )

    with pytest.raises(HTTPException) as exc:
        await family_service._ensure_invite_usable(invite)

    assert exc.value.status_code == 400
    assert exc.value.detail == "이미 처리된 가족 초대입니다."


def test_family_invite_api_hides_invite_code_when_debug_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_current_user() -> SimpleNamespace:
        return SimpleNamespace(id=1, nickname="동욱", name="홍동욱", login_id="dongwook")

    async def fake_create_family_invite(user: object, family_id: int, request: FamilyInviteCreateRequest):
        assert family_id == 10
        expires_at = datetime(2026, 5, 31, 10, 0, 0, tzinfo=family_service.config.TIMEZONE)
        return (
            SimpleNamespace(
                id=3,
                family_id=10,
                inviter_user_id=1,
                invitee_user_id=None,
                invitee_email="family@example.com",
                invitee_phone=None,
                relation_type=request.relation_type,
                member_role=request.member_role,
                status=FamilyInviteStatus.PENDING,
                expires_at=expires_at,
                used_at=None,
                created_at=datetime(2026, 5, 30, 10, 0, 0, tzinfo=family_service.config.TIMEZONE),
            ),
            "invite-token",
        )

    app.dependency_overrides[get_request_user] = fake_current_user
    monkeypatch.setattr(family_routers.config, "ENV", Env.LOCAL)
    monkeypatch.setattr(family_routers.config, "EMAIL_VERIFICATION_DEBUG", False)
    monkeypatch.setattr(family_routers.family_service, "create_family_invite", fake_create_family_invite)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/family/groups/10/invites",
                json={"invitee_email": "family@example.com", "relation_type": "SPOUSE"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert "invite_code" not in body


def test_family_invite_api_exposes_invite_code_only_when_local_debug_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_current_user() -> SimpleNamespace:
        return SimpleNamespace(id=1, nickname="동욱", name="홍동욱", login_id="dongwook")

    async def fake_create_family_invite(user: object, family_id: int, request: FamilyInviteCreateRequest):
        expires_at = datetime(2026, 5, 31, 10, 0, 0, tzinfo=family_service.config.TIMEZONE)
        return (
            SimpleNamespace(
                id=3,
                family_id=family_id,
                inviter_user_id=1,
                invitee_user_id=None,
                invitee_email="family@example.com",
                invitee_phone=None,
                relation_type=request.relation_type,
                member_role=request.member_role,
                status=FamilyInviteStatus.PENDING,
                expires_at=expires_at,
                used_at=None,
                created_at=datetime(2026, 5, 30, 10, 0, 0, tzinfo=family_service.config.TIMEZONE),
            ),
            "invite-token",
        )

    app.dependency_overrides[get_request_user] = fake_current_user
    monkeypatch.setattr(family_routers.config, "ENV", Env.LOCAL)
    monkeypatch.setattr(family_routers.config, "EMAIL_VERIFICATION_DEBUG", True)
    monkeypatch.setattr(family_routers.family_service, "create_family_invite", fake_create_family_invite)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/family/groups/10/invites",
                json={"invitee_email": "family@example.com", "relation_type": "SPOUSE"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["invite_code"] == "invite-token"
