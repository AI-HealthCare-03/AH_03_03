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


def test_family_invite_ttl_is_30_minutes() -> None:
    assert family_service.FAMILY_INVITE_TTL_MINUTES == 30


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

    async def fake_ensure_invite_target_allowed(**kwargs: object) -> None:
        assert kwargs["invitee_email"] == "family@example.com"
        assert kwargs["invitee_phone"] is None

    async def fake_generate_unique_code() -> str:
        return "12345678"

    async def fake_expire_stale_pending_invites(**kwargs: object) -> None:
        assert kwargs["invitee_email"] == "family@example.com"

    async def fake_find_existing_pending_invite(**kwargs: object) -> None:
        assert kwargs["invitee_email"] == "family@example.com"
        return None

    monkeypatch.setattr(family_service, "_ensure_owner", fake_ensure_owner)
    monkeypatch.setattr(family_service, "_ensure_invite_target_allowed", fake_ensure_invite_target_allowed)
    monkeypatch.setattr(family_service, "_generate_unique_family_invite_code", fake_generate_unique_code)
    monkeypatch.setattr(family_service, "_expire_stale_pending_invites", fake_expire_stale_pending_invites)
    monkeypatch.setattr(family_service, "_find_existing_pending_invite", fake_find_existing_pending_invite)
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
    assert invite_code == "12345678"
    assert create_payloads[0]["invitee_email"] == "family@example.com"
    assert create_payloads[0]["code_hash"] == family_service._digest("12345678")
    assert "invite_code" not in create_payloads[0]
    assert invite.expires_at - family_service._now() <= timedelta(minutes=30, seconds=1)
    assert email_jobs == [
        {
            "recipient_email": "family@example.com",
            "inviter_display_name": "동욱",
            "invite_code": "12345678",
            "invite_url": "http://localhost:8080/family?invite_code=12345678",
            "expires_at_text": invite.expires_at.astimezone(family_service.config.TIMEZONE).strftime("%Y-%m-%d %H:%M"),
        }
    ]


@pytest.mark.asyncio
async def test_family_invite_code_is_8_digit_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeInviteQuery:
        async def exists(self) -> bool:
            return False

    monkeypatch.setattr(family_service.secrets, "randbelow", lambda maximum: 1234)
    monkeypatch.setattr(family_service.FamilyInvite, "filter", staticmethod(lambda **kwargs: FakeInviteQuery()))

    invite_code = await family_service._generate_unique_family_invite_code()

    assert invite_code == "00001234"
    assert invite_code.isdigit()
    assert len(invite_code) == 8


@pytest.mark.asyncio
async def test_family_invite_code_collision_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    code_values = iter([11111111, 22222222])
    exists_values = iter([True, False])

    class FakeInviteQuery:
        async def exists(self) -> bool:
            return next(exists_values)

    monkeypatch.setattr(family_service.secrets, "randbelow", lambda maximum: next(code_values))
    monkeypatch.setattr(family_service.FamilyInvite, "filter", staticmethod(lambda **kwargs: FakeInviteQuery()))

    assert await family_service._generate_unique_family_invite_code() == "22222222"


@pytest.mark.asyncio
async def test_family_invite_code_generation_fails_after_retry_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeInviteQuery:
        async def exists(self) -> bool:
            return True

    monkeypatch.setattr(family_service.secrets, "randbelow", lambda maximum: 11111111)
    monkeypatch.setattr(family_service.FamilyInvite, "filter", staticmethod(lambda **kwargs: FakeInviteQuery()))

    with pytest.raises(HTTPException) as exc:
        await family_service._generate_unique_family_invite_code()

    assert exc.value.status_code == 400
    assert exc.value.detail == "초대 코드를 생성하지 못했습니다. 잠시 후 다시 시도해주세요."


@pytest.mark.asyncio
async def test_family_invite_email_send_is_safe_when_email_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.email_service.config.EMAIL_ENABLED", False)
    monkeypatch.setattr("app.services.email_service.config.ENV", Env.LOCAL)

    sent = await EmailService().send_family_invite_email(
        "family@example.com",
        inviter_display_name="동욱",
        invite_code="12345678",
        invite_url="http://localhost:8080/family?invite_code=12345678",
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
        invite_code="12345678",
        invite_url="http://localhost:8080/family?invite_code=12345678",
        expires_at_text="2026-05-31 10:00",
    )

    assert sent is True
    assert captured["subject"] == "[Health Ladder] 가족 건강관리 초대 안내"
    assert "가족 페이지에서 아래 초대코드를 입력해 연결을 완료해 주세요." in captured["body"]
    assert "초대 코드: 12345678" in captured["body"]
    assert "http://localhost:8080/family?invite_code=12345678" in captured["body"]
    assert "30분 동안 유효" in captured["body"]
    assert "가장 최근에 발송된 초대코드만 사용할 수 있습니다." in captured["body"]
    assert "회원가입 또는 로그인 후 가족 페이지에서 아래 초대코드를 입력해 주세요." in captured["body"]
    assert "/family/invitations/accept" not in captured["body"]
    assert '<p style="font-size: 20px; font-weight: 700;">초대 코드: 12345678</p>' in captured["html_body"]
    assert "가족 페이지에서 초대코드 입력하기" in captured["html_body"]
    assert "30분 동안 유효" in captured["html_body"]
    assert "가장 최근에 발송된 초대코드만 사용할 수 있습니다." in captured["html_body"]
    assert "회원가입 또는 로그인 후 가족 페이지에서 초대코드를 입력해 주세요." in captured["html_body"]
    assert "/family/invitations/accept" not in captured["html_body"]
    for sensitive_value in ("120", "혈압", "혈당", "체중", "질병 위험도", "OCR"):
        assert sensitive_value not in captured["body"]
        assert sensitive_value not in captured["html_body"]


@pytest.mark.asyncio
async def test_family_invite_rejects_missing_target() -> None:
    inviter = SimpleNamespace(id=1, email="me@example.com", phone_number="01011112222")

    with pytest.raises(HTTPException) as exc:
        await family_service._ensure_invite_target_allowed(
            family_id=10,
            inviter=inviter,
            invitee_email=None,
            invitee_phone=None,
            invitee_user_id=None,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "초대받을 가족의 이메일 또는 전화번호를 입력해주세요."


@pytest.mark.asyncio
async def test_family_invite_rejects_self_invite(monkeypatch: pytest.MonkeyPatch) -> None:
    inviter = SimpleNamespace(id=1, email="me@example.com", phone_number="01011112222")

    async def fake_resolve_invitee_user(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(family_service, "_resolve_invitee_user", fake_resolve_invitee_user)

    with pytest.raises(HTTPException) as exc:
        await family_service._ensure_invite_target_allowed(
            family_id=10,
            inviter=inviter,
            invitee_email="me@example.com",
            invitee_phone=None,
            invitee_user_id=None,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "본인은 가족 초대 대상이 될 수 없습니다."


@pytest.mark.asyncio
async def test_family_invite_rejects_existing_active_member(monkeypatch: pytest.MonkeyPatch) -> None:
    inviter = SimpleNamespace(id=1, email="me@example.com", phone_number="01011112222")
    invitee = SimpleNamespace(id=2)

    async def fake_resolve_invitee_user(**kwargs: object) -> SimpleNamespace:
        return invitee

    class FakeFamilyMemberQuery:
        async def exists(self) -> bool:
            return True

    monkeypatch.setattr(family_service, "_resolve_invitee_user", fake_resolve_invitee_user)
    monkeypatch.setattr(
        family_service.FamilyMember, "filter", staticmethod(lambda *args, **kwargs: FakeFamilyMemberQuery())
    )

    with pytest.raises(HTTPException) as exc:
        await family_service._ensure_invite_target_allowed(
            family_id=10,
            inviter=inviter,
            invitee_email="family@example.com",
            invitee_phone=None,
            invitee_user_id=None,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "이미 연결된 가족입니다."


@pytest.mark.asyncio
async def test_family_invite_allows_duplicate_pending_target_for_resend(monkeypatch: pytest.MonkeyPatch) -> None:
    inviter = SimpleNamespace(id=1, email="me@example.com", phone_number="01011112222")

    async def fake_resolve_invitee_user(**kwargs: object) -> None:
        return None

    class FakeFamilyMemberQuery:
        async def exists(self) -> bool:
            return False

    monkeypatch.setattr(family_service, "_resolve_invitee_user", fake_resolve_invitee_user)
    monkeypatch.setattr(
        family_service.FamilyMember, "filter", staticmethod(lambda *args, **kwargs: FakeFamilyMemberQuery())
    )

    await family_service._ensure_invite_target_allowed(
        family_id=10,
        inviter=inviter,
        invitee_email="family@example.com",
        invitee_phone=None,
        invitee_user_id=None,
    )


@pytest.mark.asyncio
async def test_family_invite_resends_existing_pending_email_invite(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_now = datetime(2026, 5, 30, 10, 0, 0, tzinfo=family_service.config.TIMEZONE)
    email_jobs: list[dict[str, str]] = []
    inviter = SimpleNamespace(id=1, nickname="동욱", name="홍동욱", login_id="dongwook")
    family = SimpleNamespace(id=10)
    existing_invite = SimpleNamespace(
        id=7,
        family_id=10,
        inviter_user_id=1,
        invitee_user_id=None,
        invitee_email="family@example.com",
        invitee_phone=None,
        code_hash=family_service._digest("11111111"),
        relation_type=FamilyRelationType.OTHER,
        member_role=FamilyMemberRole.MEMBER,
        status=FamilyInviteStatus.PENDING,
        expires_at=fixed_now + timedelta(minutes=3),
        used_at=None,
        created_at=fixed_now - timedelta(minutes=1),
        saved_fields=[],
    )

    async def fake_save(update_fields: list[str]) -> None:
        existing_invite.saved_fields = update_fields

    existing_invite.save = fake_save

    async def fake_ensure_owner(user: object, family_id: int) -> SimpleNamespace:
        assert user is inviter
        assert family_id == 10
        return family

    async def fake_ensure_invite_target_allowed(**kwargs: object) -> None:
        assert kwargs["invitee_email"] == "family@example.com"

    async def fake_generate_unique_code() -> str:
        return "22222222"

    async def fake_expire_stale_pending_invites(**kwargs: object) -> None:
        assert kwargs["now"] == fixed_now

    async def fake_find_existing_pending_invite(**kwargs: object) -> SimpleNamespace:
        assert kwargs["invitee_email"] == "family@example.com"
        return existing_invite

    async def fake_enqueue_family_invite_email_send(**kwargs: str) -> None:
        email_jobs.append(kwargs)

    monkeypatch.setattr(family_service, "_now", lambda: fixed_now)
    monkeypatch.setattr(family_service, "_ensure_owner", fake_ensure_owner)
    monkeypatch.setattr(family_service, "_ensure_invite_target_allowed", fake_ensure_invite_target_allowed)
    monkeypatch.setattr(family_service, "_generate_unique_family_invite_code", fake_generate_unique_code)
    monkeypatch.setattr(family_service, "_expire_stale_pending_invites", fake_expire_stale_pending_invites)
    monkeypatch.setattr(family_service, "_find_existing_pending_invite", fake_find_existing_pending_invite)
    monkeypatch.setattr(
        family_service.service_jobs,
        "enqueue_family_invite_email_send",
        fake_enqueue_family_invite_email_send,
    )

    invite, invite_code = await family_service.create_family_invite(
        inviter,
        10,
        FamilyInviteCreateRequest(
            invitee_email="family@example.com",
            relation_type=FamilyRelationType.SPOUSE,
            member_role=FamilyMemberRole.GUARDIAN,
        ),
    )

    assert invite is existing_invite
    assert invite_code == "22222222"
    assert existing_invite.code_hash != family_service._digest("11111111")
    assert existing_invite.code_hash == family_service._digest("22222222")
    assert existing_invite.expires_at == fixed_now + timedelta(minutes=30)
    assert existing_invite.relation_type == FamilyRelationType.SPOUSE
    assert existing_invite.member_role == FamilyMemberRole.GUARDIAN
    assert "code_hash" in existing_invite.saved_fields
    assert "expires_at" in existing_invite.saved_fields
    assert email_jobs[0]["invite_code"] == "22222222"
    assert email_jobs[0]["invite_url"].endswith("/family?invite_code=22222222")


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


@pytest.mark.asyncio
async def test_family_invite_preview_returns_invite_info_without_accepting(monkeypatch: pytest.MonkeyPatch) -> None:
    user = SimpleNamespace(id=2, email="family@example.com", phone_number=None)
    invite = SimpleNamespace(
        id=9,
        family_id=10,
        family=SimpleNamespace(id=10, name="우리 가족"),
        inviter_user=SimpleNamespace(id=1, nickname="동욱", name="홍동욱", login_id="dongwook"),
        invitee_user_id=None,
        invitee_email="family@example.com",
        invitee_phone=None,
        status=FamilyInviteStatus.PENDING,
        used_at=None,
        expires_at=family_service._now() + timedelta(minutes=5),
    )
    calls: list[str] = []

    class FakeInviteQuery:
        def select_related(self, *args: str) -> FakeInviteQuery:
            calls.extend(args)
            return self

        async def first(self) -> SimpleNamespace:
            return invite

    monkeypatch.setattr(family_service.FamilyInvite, "filter", staticmethod(lambda **kwargs: FakeInviteQuery()))

    preview = await family_service.preview_family_invite_by_code(user, "12345678")

    assert calls == ["family", "inviter_user"]
    assert preview.invite_id == 9
    assert preview.family_id == 10
    assert preview.family_name == "우리 가족"
    assert preview.inviter_display_name == "동욱"
    assert preview.invitee_email == "family@example.com"
    assert preview.status == FamilyInviteStatus.PENDING
    assert invite.status == FamilyInviteStatus.PENDING
    assert invite.used_at is None


@pytest.mark.asyncio
async def test_family_invite_preview_rejects_wrong_target(monkeypatch: pytest.MonkeyPatch) -> None:
    user = SimpleNamespace(id=2, email="other@example.com", phone_number=None)
    invite = SimpleNamespace(
        id=9,
        family_id=10,
        family=SimpleNamespace(id=10, name="우리 가족"),
        inviter_user=SimpleNamespace(id=1, nickname="동욱", name="홍동욱", login_id="dongwook"),
        invitee_user_id=None,
        invitee_email="family@example.com",
        invitee_phone=None,
        status=FamilyInviteStatus.PENDING,
        used_at=None,
        expires_at=family_service._now() + timedelta(minutes=5),
    )

    class FakeInviteQuery:
        def select_related(self, *args: str) -> FakeInviteQuery:
            return self

        async def first(self) -> SimpleNamespace:
            return invite

    monkeypatch.setattr(family_service.FamilyInvite, "filter", staticmethod(lambda **kwargs: FakeInviteQuery()))

    with pytest.raises(HTTPException) as exc:
        await family_service.preview_family_invite_by_code(user, "12345678")

    assert exc.value.status_code == 403
    assert exc.value.detail == "이 초대 코드를 사용할 수 없습니다."


@pytest.mark.asyncio
async def test_list_family_invites_requires_member_and_orders_recent_first(monkeypatch: pytest.MonkeyPatch) -> None:
    user = SimpleNamespace(id=1)
    invites = [SimpleNamespace(id=2), SimpleNamespace(id=1)]
    captured: dict[str, object] = {}

    async def fake_ensure_active_member(request_user: object, family_id: int) -> SimpleNamespace:
        captured["user"] = request_user
        captured["family_id"] = family_id
        return SimpleNamespace(id=7)

    class FakeInviteQuery:
        def order_by(self, *fields: str) -> FakeInviteQuery:
            captured["order_by"] = fields
            return self

        def __await__(self):
            async def _result() -> list[SimpleNamespace]:
                return invites

            return _result().__await__()

    monkeypatch.setattr(family_service, "_ensure_active_member", fake_ensure_active_member)
    monkeypatch.setattr(family_service.FamilyInvite, "filter", staticmethod(lambda **kwargs: FakeInviteQuery()))

    result = await family_service.list_family_invites(user, 10)

    assert result == invites
    assert captured["user"] is user
    assert captured["family_id"] == 10
    assert captured["order_by"] == ("-created_at", "-id")


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
            "12345678",
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


def test_family_invite_list_api_does_not_expose_code_or_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_current_user() -> SimpleNamespace:
        return SimpleNamespace(id=1, nickname="동욱", name="홍동욱", login_id="dongwook")

    async def fake_list_family_invites(user: object, family_id: int) -> list[SimpleNamespace]:
        assert family_id == 10
        expires_at = datetime(2026, 5, 31, 10, 0, 0, tzinfo=family_service.config.TIMEZONE)
        return [
            SimpleNamespace(
                id=3,
                family_id=10,
                inviter_user_id=1,
                invitee_user_id=None,
                invitee_email="family@example.com",
                invitee_phone=None,
                relation_type=FamilyRelationType.SPOUSE,
                member_role=FamilyMemberRole.MEMBER,
                status=FamilyInviteStatus.PENDING,
                expires_at=expires_at,
                used_at=None,
                created_at=datetime(2026, 5, 30, 10, 0, 0, tzinfo=family_service.config.TIMEZONE),
                invite_code="12345678",
                code_hash="secret-hash",
            )
        ]

    app.dependency_overrides[get_request_user] = fake_current_user
    monkeypatch.setattr(family_routers.family_service, "list_family_invites", fake_list_family_invites)
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/family/groups/10/invites")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()[0]
    assert body["invitee_email"] == "family@example.com"
    assert body["status"] == "PENDING"
    assert "invite_code" not in body
    assert "code_hash" not in body


def test_family_invite_preview_api_does_not_expose_code_or_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_current_user() -> SimpleNamespace:
        return SimpleNamespace(id=2, email="family@example.com", nickname="가족")

    async def fake_preview_family_invite_by_code(user: object, code: str) -> family_service.FamilyInvitePreview:
        assert code == "12345678"
        return family_service.FamilyInvitePreview(
            invite_id=3,
            family_id=10,
            family_name="우리 가족",
            inviter_display_name="동욱",
            invitee_email="family@example.com",
            status=FamilyInviteStatus.PENDING,
            expires_at=datetime(2026, 5, 31, 10, 0, 0, tzinfo=family_service.config.TIMEZONE),
        )

    app.dependency_overrides[get_request_user] = fake_current_user
    monkeypatch.setattr(
        family_routers.family_service, "preview_family_invite_by_code", fake_preview_family_invite_by_code
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/family/invites/code/preview", json={"invite_code": "12345678"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["invite_id"] == 3
    assert body["family_name"] == "우리 가족"
    assert body["inviter_display_name"] == "동욱"
    assert "invite_code" not in body
    assert "code_hash" not in body


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
            "12345678",
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
    assert response.json()["invite_code"] == "12345678"
