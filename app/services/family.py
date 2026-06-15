import hashlib
import secrets
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import reduce
from math import ceil
from operator import or_

from fastapi import HTTPException
from starlette import status
from tortoise.expressions import Q
from tortoise.transactions import in_transaction

from app.core import config
from app.core.utils.common import normalize_phone_number
from app.dtos.family import (
    FamilyActionShareSettingUpdateRequest,
    FamilyGroupCreateRequest,
    FamilyGroupUpdateRequest,
    FamilyInviteCreateRequest,
    FamilyMemberCreateUnregisteredRequest,
    FamilyNotificationSettingUpdateRequest,
    FamilyShareSettingUpdateRequest,
)
from app.dtos.notifications import NotificationCreateRequest
from app.models.family import (
    Family,
    FamilyInvite,
    FamilyInviteStatus,
    FamilyMember,
    FamilyMemberRole,
    FamilyMemberStatus,
    FamilyNotificationSetting,
    FamilyShareSetting,
    FamilyStatus,
)
from app.models.notifications import Notification
from app.models.users import User
from app.services import notifications as notification_service
from app.services import service_jobs
from app.services.email_service import EmailService

FAMILY_INVITE_TTL_HOURS = 24
FAMILY_INVITE_CODE_LENGTH = 8
FAMILY_INVITE_CODE_MAX_ATTEMPTS = 10
FAMILY_CHALLENGE_COMPLETED_RELATED_TYPE = "family_challenge_completed"
FAMILY_CHALLENGE_MISSED_RELATED_TYPE = "family_challenge_missed"


@dataclass(frozen=True)
class FamilyChallengeAlertContext:
    owner_user_id: int
    owner_display_name: str
    user_challenge_id: int


def _normalize_email(email: str | None) -> str | None:
    return email.strip().lower() if email else None


def _normalize_phone(phone_number: str | None) -> str | None:
    return normalize_phone_number(phone_number) if phone_number else None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(config.TIMEZONE)


def _forbidden(message: str = "가족 정보에 접근할 권한이 없습니다.") -> None:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=message)


def _not_found(message: str = "가족 정보를 찾을 수 없습니다.") -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)


def _bad_request(message: str) -> None:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def _display_name(user: User) -> str:
    return user.nickname or user.name or user.login_id or "회원"


def _family_invite_accept_url(invite_code: str) -> str:
    base_url = config.FRONTEND_BASE_URL.rstrip("/")
    encoded_code = urllib.parse.quote(invite_code, safe="")
    return f"{base_url}/family?invite_code={encoded_code}"


def _format_invite_expiration(expires_at: datetime) -> str:
    return expires_at.astimezone(config.TIMEZONE).strftime("%Y-%m-%d %H:%M")


async def _resolve_invite_recipient_email(payload: FamilyInviteCreateRequest) -> str | None:
    if payload.invitee_email:
        return _normalize_email(str(payload.invitee_email))
    if payload.invitee_user_id is None:
        return None

    invitee = await User.get_or_none(id=payload.invitee_user_id)
    if invitee is None:
        _bad_request("초대받는 사용자를 찾을 수 없습니다.")
    return _normalize_email(invitee.email)


def _generate_family_invite_code() -> str:
    return f"{secrets.randbelow(10**FAMILY_INVITE_CODE_LENGTH):0{FAMILY_INVITE_CODE_LENGTH}d}"


async def _generate_unique_family_invite_code() -> str:
    for _ in range(FAMILY_INVITE_CODE_MAX_ATTEMPTS):
        invite_code = _generate_family_invite_code()
        exists = await FamilyInvite.filter(code_hash=_digest(invite_code)).exists()
        if not exists:
            return invite_code
    _bad_request("초대 코드를 생성하지 못했습니다. 잠시 후 다시 시도해주세요.")


async def _resolve_invitee_user(
    *,
    invitee_email: str | None,
    invitee_phone: str | None,
    invitee_user_id: int | None,
) -> User | None:
    if invitee_user_id is not None:
        invitee = await User.get_or_none(id=invitee_user_id)
        if invitee is None:
            _bad_request("초대받는 사용자를 찾을 수 없습니다.")
        return invitee
    if invitee_email:
        return await User.get_or_none(email=invitee_email)
    if invitee_phone:
        return await User.get_or_none(phone_number=invitee_phone)
    return None


async def _ensure_invite_target_allowed(
    *,
    family_id: int,
    inviter: User,
    invitee_email: str | None,
    invitee_phone: str | None,
    invitee_user_id: int | None,
) -> None:
    if not any((invitee_email, invitee_phone, invitee_user_id)):
        _bad_request("초대받을 가족의 이메일 또는 전화번호를 입력해주세요.")

    invitee_user = await _resolve_invitee_user(
        invitee_email=invitee_email,
        invitee_phone=invitee_phone,
        invitee_user_id=invitee_user_id,
    )
    inviter_email = _normalize_email(inviter.email)
    inviter_phone = _normalize_phone(inviter.phone_number)
    if (
        (invitee_user is not None and int(invitee_user.id) == int(inviter.id))
        or (invitee_email is not None and invitee_email == inviter_email)
        or (invitee_phone is not None and invitee_phone == inviter_phone)
    ):
        _bad_request("본인은 가족 초대 대상이 될 수 없습니다.")

    active_filters: list[Q] = []
    if invitee_user is not None:
        active_filters.append(Q(user_id=invitee_user.id))
    if invitee_email:
        active_filters.append(Q(email=invitee_email))
    if invitee_phone:
        active_filters.append(Q(phone_number=invitee_phone))
    if (
        active_filters
        and await FamilyMember.filter(
            reduce(or_, active_filters),
            family_id=family_id,
            status=FamilyMemberStatus.ACTIVE,
        ).exists()
    ):
        _bad_request("이미 연결된 가족입니다.")

    pending_query = FamilyInvite.filter(family_id=family_id, status=FamilyInviteStatus.PENDING, expires_at__gte=_now())
    if invitee_user_id is not None and await pending_query.filter(invitee_user_id=invitee_user_id).exists():
        _bad_request("이미 대기 중인 가족 초대가 있습니다.")
    if invitee_email and await pending_query.filter(invitee_email=invitee_email).exists():
        _bad_request("이미 대기 중인 가족 초대가 있습니다.")
    if invitee_phone and await pending_query.filter(invitee_phone=invitee_phone).exists():
        _bad_request("이미 대기 중인 가족 초대가 있습니다.")


async def _send_family_invite_email(
    *,
    recipient_email: str | None,
    inviter: User,
    invite_code: str,
    expires_at: datetime,
) -> bool:
    if recipient_email is None:
        return False
    return await EmailService().send_family_invite_email(
        recipient_email,
        inviter_display_name=_display_name(inviter),
        invite_code=invite_code,
        invite_url=_family_invite_accept_url(invite_code),
        expires_at_text=_format_invite_expiration(expires_at),
    )


async def _get_active_member(user_id: int, family_id: int) -> FamilyMember | None:
    return await FamilyMember.filter(
        family_id=family_id,
        user_id=user_id,
        status=FamilyMemberStatus.ACTIVE,
        is_registered=True,
    ).first()


async def _ensure_active_member(user: User, family_id: int) -> FamilyMember:
    family = await Family.filter(id=family_id, status=FamilyStatus.ACTIVE).first()
    if family is None:
        _not_found()
    member = await _get_active_member(int(user.id), family_id)
    if member is None:
        _forbidden()
    return member


async def _ensure_owner(user: User, family_id: int) -> Family:
    family = await Family.filter(id=family_id, status=FamilyStatus.ACTIVE).first()
    if family is None:
        _not_found()

    member = await _get_active_member(int(user.id), family_id)
    if int(family.owner_user_id) != int(user.id) and (member is None or member.member_role != FamilyMemberRole.OWNER):
        _forbidden("가족 그룹 소유자만 처리할 수 있습니다.")
    return family


async def _find_common_active_family_id(owner_user_id: int, family_user_id: int) -> int:
    owner_family_ids = await FamilyMember.filter(
        user_id=owner_user_id,
        status=FamilyMemberStatus.ACTIVE,
        is_registered=True,
        family__status=FamilyStatus.ACTIVE,
    ).values_list("family_id", flat=True)
    member = await FamilyMember.filter(
        user_id=family_user_id,
        status=FamilyMemberStatus.ACTIVE,
        is_registered=True,
        family_id__in=list(owner_family_ids),
    ).first()
    if member is None:
        _forbidden("연결된 가족 사용자만 설정할 수 있습니다.")
    return int(member.family_id)


def _invite_matches_user(invite: FamilyInvite, user: User, *, allow_open_code: bool) -> bool:
    if invite.invitee_user_id is not None:
        return int(invite.invitee_user_id) == int(user.id)

    if invite.invitee_email:
        return invite.invitee_email == _normalize_email(user.email)

    if invite.invitee_phone:
        return invite.invitee_phone == _normalize_phone(user.phone_number)

    return allow_open_code


async def _ensure_invite_usable(invite: FamilyInvite) -> None:
    now = _now()
    if invite.status != FamilyInviteStatus.PENDING or invite.used_at is not None:
        _bad_request("이미 처리된 가족 초대입니다.")
    if invite.expires_at < now:
        await FamilyInvite.filter(id=invite.id).update(status=FamilyInviteStatus.EXPIRED)
        _bad_request("만료된 가족 초대입니다.")


async def _create_default_share_settings(family_id: int, new_user_id: int) -> None:
    members = await FamilyMember.filter(
        family_id=family_id,
        status=FamilyMemberStatus.ACTIVE,
        is_registered=True,
    ).exclude(user_id=None)

    for member in members:
        member_user_id = int(member.user_id)
        if member_user_id == int(new_user_id):
            continue
        await FamilyShareSetting.get_or_create(
            family_id=family_id,
            owner_user_id=new_user_id,
            viewer_user_id=member_user_id,
        )
        await FamilyShareSetting.get_or_create(
            family_id=family_id,
            owner_user_id=member_user_id,
            viewer_user_id=new_user_id,
        )


async def create_family_group(user: User, payload: FamilyGroupCreateRequest) -> Family:
    async with in_transaction() as connection:
        family = await Family.create(
            name=payload.name.strip(),
            owner_user_id=user.id,
            using_db=connection,
        )
        await FamilyMember.create(
            family=family,
            user_id=user.id,
            display_name=_display_name(user),
            phone_number=_normalize_phone(user.phone_number),
            email=_normalize_email(user.email),
            relation_type="SELF",
            member_role=FamilyMemberRole.OWNER,
            status=FamilyMemberStatus.ACTIVE,
            is_registered=True,
            using_db=connection,
        )
    return family


async def list_my_family_groups(user: User) -> list[Family]:
    return (
        await Family.filter(
            status=FamilyStatus.ACTIVE,
            members__user_id=user.id,
            members__status=FamilyMemberStatus.ACTIVE,
        )
        .distinct()
        .order_by("-created_at")
    )


async def get_family_group_detail(user: User, family_id: int) -> tuple[Family, list[FamilyMember]]:
    await _ensure_active_member(user, family_id)
    family = await Family.get(id=family_id)
    members = await FamilyMember.filter(family_id=family_id).order_by("id")
    return family, members


async def update_family_group(user: User, family_id: int, payload: FamilyGroupUpdateRequest) -> Family:
    family = await _ensure_owner(user, family_id)
    data = payload.model_dump(exclude_unset=True)
    if "name" in data and data["name"] is not None:
        family.name = data["name"].strip()
        await family.save(update_fields=["name", "updated_at"])
    return family


async def remove_family_group(user: User, family_id: int) -> None:
    family = await _ensure_owner(user, family_id)
    await Family.filter(id=family.id).update(status=FamilyStatus.REMOVED)
    await FamilyMember.filter(family_id=family.id).update(status=FamilyMemberStatus.REMOVED)
    await FamilyInvite.filter(family_id=family.id, status=FamilyInviteStatus.PENDING).update(
        status=FamilyInviteStatus.CANCELED
    )


async def list_family_members(user: User, family_id: int) -> list[FamilyMember]:
    await _ensure_active_member(user, family_id)
    return await FamilyMember.filter(family_id=family_id).order_by("id")


async def add_unregistered_family_member(
    user: User,
    family_id: int,
    payload: FamilyMemberCreateUnregisteredRequest,
) -> FamilyMember:
    family = await _ensure_owner(user, family_id)
    return await FamilyMember.create(
        family=family,
        user_id=None,
        display_name=payload.display_name.strip(),
        phone_number=_normalize_phone(payload.phone_number),
        email=_normalize_email(str(payload.email)) if payload.email else None,
        relation_type=payload.relation_type,
        member_role=FamilyMemberRole.DEPENDENT,
        status=FamilyMemberStatus.PENDING_UNREGISTERED,
        is_registered=False,
    )


async def remove_family_member(user: User, member_id: int) -> None:
    member = await FamilyMember.get_or_none(id=member_id)
    if member is None:
        _not_found("가족 구성원을 찾을 수 없습니다.")

    can_remove = member.user_id is not None and int(member.user_id) == int(user.id)
    family = await Family.get_or_none(id=member.family_id)
    if family and int(family.owner_user_id) == int(user.id):
        can_remove = True

    if not can_remove:
        _forbidden()

    await FamilyMember.filter(id=member.id).update(status=FamilyMemberStatus.REMOVED)
    if member.user_id is not None:
        await FamilyShareSetting.filter(
            Q(owner_user_id=member.user_id) | Q(viewer_user_id=member.user_id),
            family_id=member.family_id,
        ).update(
            share_health_records=False,
            share_analysis_results=False,
            share_diet_records=False,
            share_medications=False,
            share_challenges=False,
            share_exam_reports=False,
            receive_analysis_alerts=False,
            receive_abnormal_value_alerts=False,
            receive_medication_alerts=False,
        )


async def create_family_invite(
    user: User,
    family_id: int,
    payload: FamilyInviteCreateRequest,
) -> tuple[FamilyInvite, str]:
    family = await _ensure_owner(user, family_id)
    expires_at = _now() + timedelta(hours=FAMILY_INVITE_TTL_HOURS)
    invitee_email = await _resolve_invite_recipient_email(payload)
    invitee_phone = _normalize_phone(payload.invitee_phone)
    await _ensure_invite_target_allowed(
        family_id=family_id,
        inviter=user,
        invitee_email=invitee_email,
        invitee_phone=invitee_phone,
        invitee_user_id=payload.invitee_user_id,
    )
    invite_code = await _generate_unique_family_invite_code()
    invite = await FamilyInvite.create(
        family=family,
        inviter_user_id=user.id,
        invitee_user_id=payload.invitee_user_id,
        invitee_email=invitee_email,
        invitee_phone=invitee_phone,
        code_hash=_digest(invite_code),
        relation_type=payload.relation_type,
        member_role=payload.member_role,
        expires_at=expires_at,
        status=FamilyInviteStatus.PENDING,
    )
    if invitee_email:
        await service_jobs.enqueue_family_invite_email_send(
            recipient_email=invitee_email,
            inviter_display_name=_display_name(user),
            invite_code=invite_code,
            invite_url=_family_invite_accept_url(invite_code),
            expires_at_text=_format_invite_expiration(expires_at),
        )
    return invite, invite_code


async def list_my_family_invites(user: User) -> list[FamilyInvite]:
    return await FamilyInvite.filter(
        Q(invitee_user_id=user.id)
        | Q(invitee_email=_normalize_email(user.email))
        | Q(invitee_phone=_normalize_phone(user.phone_number)),
        status=FamilyInviteStatus.PENDING,
        expires_at__gte=_now(),
    ).order_by("-created_at")


async def accept_family_invite(user: User, invite_id: int) -> FamilyMember:
    invite = await FamilyInvite.get_or_none(id=invite_id)
    if invite is None:
        _not_found("가족 초대를 찾을 수 없습니다.")
    if not _invite_matches_user(invite, user, allow_open_code=False):
        _forbidden("본인에게 발송된 초대만 수락할 수 있습니다.")
    return await _accept_invite(user, invite)


async def decline_family_invite(user: User, invite_id: int) -> FamilyInvite:
    invite = await FamilyInvite.get_or_none(id=invite_id)
    if invite is None:
        _not_found("가족 초대를 찾을 수 없습니다.")
    if not _invite_matches_user(invite, user, allow_open_code=False):
        _forbidden("본인에게 발송된 초대만 거절할 수 있습니다.")

    await _ensure_invite_usable(invite)
    invite.status = FamilyInviteStatus.DECLINED
    await invite.save(update_fields=["status"])
    return invite


async def accept_family_invite_by_code(user: User, code: str) -> FamilyMember:
    invite = await FamilyInvite.get_or_none(code_hash=_digest(code.strip()))
    if invite is None:
        _not_found("가족 초대를 찾을 수 없습니다.")
    if not _invite_matches_user(invite, user, allow_open_code=True):
        _forbidden("이 초대 코드를 사용할 수 없습니다.")
    return await _accept_invite(user, invite)


async def _accept_invite(user: User, invite: FamilyInvite) -> FamilyMember:
    await _ensure_invite_usable(invite)

    existing = await FamilyMember.filter(family_id=invite.family_id, user_id=user.id).first()
    if existing is not None and existing.status == FamilyMemberStatus.ACTIVE:
        _bad_request("이미 연결된 가족 그룹입니다.")

    async with in_transaction():
        if existing is None:
            member = await FamilyMember.create(
                family_id=invite.family_id,
                user_id=user.id,
                display_name=_display_name(user),
                phone_number=_normalize_phone(user.phone_number),
                email=_normalize_email(user.email),
                relation_type=invite.relation_type,
                member_role=invite.member_role,
                status=FamilyMemberStatus.ACTIVE,
                is_registered=True,
            )
        else:
            await FamilyMember.filter(id=existing.id).update(
                display_name=_display_name(user),
                phone_number=_normalize_phone(user.phone_number),
                email=_normalize_email(user.email),
                relation_type=invite.relation_type,
                member_role=invite.member_role,
                status=FamilyMemberStatus.ACTIVE,
                is_registered=True,
                user_id=user.id,
            )
            member = await FamilyMember.get(id=existing.id)

        now = _now()
        await FamilyInvite.filter(id=invite.id).update(
            status=FamilyInviteStatus.ACCEPTED,
            used_at=now,
        )
        await _create_default_share_settings(invite.family_id, int(user.id))
    return member


async def list_family_share_settings(user: User, family_id: int | None = None) -> list[FamilyShareSetting]:
    query = FamilyShareSetting.filter(Q(owner_user_id=user.id) | Q(viewer_user_id=user.id))
    if family_id is not None:
        await _ensure_active_member(user, family_id)
        query = query.filter(family_id=family_id)
    return await query.order_by("-created_at")


async def update_family_share_setting(
    user: User,
    setting_id: int,
    payload: FamilyShareSettingUpdateRequest,
) -> FamilyShareSetting:
    setting = await FamilyShareSetting.get_or_none(id=setting_id)
    if setting is None:
        _not_found("가족 공유 설정을 찾을 수 없습니다.")
    if int(setting.owner_user_id) != int(user.id):
        _forbidden("공유 권한은 정보 소유자만 변경할 수 있습니다.")

    data = payload.model_dump(exclude_unset=True)
    if data:
        await FamilyShareSetting.filter(id=setting.id).update(**data)
        setting = await FamilyShareSetting.get(id=setting.id)
    return setting


async def get_family_action_share_setting(user: User, family_user_id: int) -> FamilyShareSetting:
    family_id = await _find_common_active_family_id(int(user.id), int(family_user_id))
    setting, _ = await FamilyShareSetting.get_or_create(
        family_id=family_id,
        owner_user_id=user.id,
        viewer_user_id=family_user_id,
    )
    return setting


async def update_family_action_share_setting(
    user: User,
    family_user_id: int,
    payload: FamilyActionShareSettingUpdateRequest,
) -> FamilyShareSetting:
    setting = await get_family_action_share_setting(user, family_user_id)
    data = payload.model_dump(exclude_unset=True)
    if data:
        await FamilyShareSetting.filter(id=setting.id).update(**data)
        setting = await FamilyShareSetting.get(id=setting.id)
    return setting


async def get_family_notification_setting(user: User, owner_user_id: int) -> FamilyNotificationSetting:
    await _find_common_active_family_id(int(owner_user_id), int(user.id))
    setting, _ = await FamilyNotificationSetting.get_or_create(
        owner_user_id=owner_user_id,
        family_user_id=user.id,
    )
    return setting


async def update_family_notification_setting(
    user: User,
    owner_user_id: int,
    payload: FamilyNotificationSettingUpdateRequest,
) -> FamilyNotificationSetting:
    setting = await get_family_notification_setting(user, owner_user_id)
    data = payload.model_dump(exclude_unset=True)
    if data:
        await FamilyNotificationSetting.filter(id=setting.id).update(**data)
        setting = await FamilyNotificationSetting.get(id=setting.id)
    return setting


async def _create_family_alerts(
    *,
    context: FamilyChallengeAlertContext,
    share_field: str,
    notify_field: str,
    related_type: str,
    title: str,
    message: str,
) -> list[Notification]:
    share_settings = await FamilyShareSetting.filter(owner_user_id=context.owner_user_id, **{share_field: True})
    created: list[Notification] = []
    for share_setting in share_settings:
        family_user_id = int(share_setting.viewer_user_id)
        notification_setting = await FamilyNotificationSetting.get_or_none(
            owner_user_id=context.owner_user_id,
            family_user_id=family_user_id,
        )
        if notification_setting is None or not getattr(notification_setting, notify_field):
            continue
        if not notification_setting.channel_in_app:
            continue

        existing = await Notification.filter(
            user_id=family_user_id,
            notification_type="FAMILY_ALERT",
            related_type=related_type,
            related_id=context.user_challenge_id,
        ).exists()
        if existing:
            continue

        notification = await notification_service.create_notification(
            family_user_id,
            NotificationCreateRequest(
                notification_type="FAMILY_ALERT",
                title=title,
                message=message,
                related_type=related_type,
                related_id=context.user_challenge_id,
            ),
        )
        created.append(notification)
    return created


async def _get_completed_challenge_alert_context(user_challenge_id: int) -> FamilyChallengeAlertContext | None:
    from app.models.challenges import ChallengeLog, UserChallenge, UserChallengeStatus
    from app.services import challenges as challenge_service

    user_challenge = await UserChallenge.get_or_none(id=user_challenge_id).select_related("challenge", "user")
    if user_challenge is None:
        return None
    if user_challenge.status == UserChallengeStatus.CANCELED or user_challenge.canceled_at is not None:
        return None

    challenge = user_challenge.challenge
    duration_days = challenge.duration_days if challenge and challenge.duration_days > 0 else 7
    required_days = max(1, ceil(duration_days * 0.8))
    expected_done_at = user_challenge.expected_done_at or user_challenge.started_at + timedelta(days=duration_days)
    if expected_done_at > _now():
        return None

    daily_goal_count = challenge_service._get_daily_goal_count(challenge)
    completed_dates = [
        completed_date
        for completed_date in [
            challenge_service._to_kst_date(log.completed_at)
            for log in await ChallengeLog.filter(user_challenge_id=user_challenge.id, is_completed=True).only(
                "completed_at"
            )
        ]
        if completed_date is not None
    ]
    if challenge_service._count_completed_days(completed_dates, daily_goal_count) < required_days:
        return None

    return FamilyChallengeAlertContext(
        owner_user_id=int(user_challenge.user_id),
        owner_display_name=_display_name(user_challenge.user),
        user_challenge_id=int(user_challenge.id),
    )


async def notify_family_challenge_completed(user_challenge_id: int) -> list[Notification]:
    context = await _get_completed_challenge_alert_context(user_challenge_id)
    if context is None:
        return []
    message = f"{context.owner_display_name}님이 이번 챌린지 목표를 달성했어요."
    return await _create_family_alerts(
        context=context,
        share_field="share_challenge_status",
        notify_field="notify_challenge_completed",
        related_type=FAMILY_CHALLENGE_COMPLETED_RELATED_TYPE,
        title="가족 챌린지 알림",
        message=message,
    )


async def notify_family_challenge_missed(user_challenge_id: int) -> list[Notification]:
    # TODO: 스케줄러에서 7일 기간 종료 후 미기록/미달성 챌린지를 찾아 호출한다.
    from app.models.challenges import UserChallenge, UserChallengeStatus

    user_challenge = await UserChallenge.get_or_none(id=user_challenge_id).select_related("user")
    if user_challenge is None:
        return []
    if user_challenge.status == UserChallengeStatus.CANCELED or user_challenge.canceled_at is not None:
        return []

    context = FamilyChallengeAlertContext(
        owner_user_id=int(user_challenge.user_id),
        owner_display_name=_display_name(user_challenge.user),
        user_challenge_id=int(user_challenge.id),
    )
    message = f"{context.owner_display_name}님이 오늘 챌린지를 아직 기록하지 않았어요."
    return await _create_family_alerts(
        context=context,
        share_field="share_challenge_status",
        notify_field="notify_challenge_missed",
        related_type=FAMILY_CHALLENGE_MISSED_RELATED_TYPE,
        title="가족 챌린지 알림",
        message=message,
    )
