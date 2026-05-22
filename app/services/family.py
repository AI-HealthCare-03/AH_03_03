import hashlib
import secrets
from datetime import datetime, timedelta

from fastapi import HTTPException
from starlette import status
from tortoise.expressions import Q
from tortoise.transactions import in_transaction

from app.core import config
from app.core.utils.common import normalize_phone_number
from app.dtos.family import (
    FamilyGroupCreateRequest,
    FamilyGroupUpdateRequest,
    FamilyInviteCreateRequest,
    FamilyMemberCreateUnregisteredRequest,
    FamilyShareSettingUpdateRequest,
)
from app.models.family import (
    Family,
    FamilyInvite,
    FamilyInviteStatus,
    FamilyMember,
    FamilyMemberRole,
    FamilyMemberStatus,
    FamilyShareSetting,
    FamilyStatus,
)
from app.models.users import User

FAMILY_INVITE_TTL_HOURS = 24


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
            display_name=user.name,
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
    invite_code = secrets.token_urlsafe(32)
    expires_at = _now() + timedelta(hours=FAMILY_INVITE_TTL_HOURS)
    invite = await FamilyInvite.create(
        family=family,
        inviter_user_id=user.id,
        invitee_user_id=payload.invitee_user_id,
        invitee_email=_normalize_email(str(payload.invitee_email)) if payload.invitee_email else None,
        invitee_phone=_normalize_phone(payload.invitee_phone),
        code_hash=_digest(invite_code),
        relation_type=payload.relation_type,
        member_role=payload.member_role,
        expires_at=expires_at,
        status=FamilyInviteStatus.PENDING,
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
                display_name=user.name,
                phone_number=_normalize_phone(user.phone_number),
                email=_normalize_email(user.email),
                relation_type=invite.relation_type,
                member_role=invite.member_role,
                status=FamilyMemberStatus.ACTIVE,
                is_registered=True,
            )
        else:
            await FamilyMember.filter(id=existing.id).update(
                display_name=user.name,
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
