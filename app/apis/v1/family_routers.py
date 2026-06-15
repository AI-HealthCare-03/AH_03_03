from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import get_request_user
from app.core import config
from app.dtos.family import (
    FamilyActionShareSettingUpdateRequest,
    FamilyGroupCreateRequest,
    FamilyGroupDetailResponse,
    FamilyGroupResponse,
    FamilyGroupUpdateRequest,
    FamilyInviteAcceptCodeRequest,
    FamilyInviteCreateRequest,
    FamilyInviteResponse,
    FamilyMemberCreateUnregisteredRequest,
    FamilyMemberResponse,
    FamilyNotificationSettingResponse,
    FamilyNotificationSettingUpdateRequest,
    FamilySentInviteResponse,
    FamilyShareSettingResponse,
    FamilyShareSettingUpdateRequest,
)
from app.models.family import Family, FamilyInvite, FamilyMember, FamilyNotificationSetting, FamilyShareSetting
from app.models.users import User
from app.services import family as family_service

family_router = APIRouter(prefix="/family", tags=["family"])


def _allow_family_invite_debug_response() -> bool:
    # 초대코드는 이메일 인증코드와 같은 민감한 일회성 값이므로 local/demo 디버그에서만 응답에 노출한다.
    return config.EMAIL_VERIFICATION_DEBUG and not config.is_production


def _family_response(family: Family) -> FamilyGroupResponse:
    return FamilyGroupResponse.model_validate(family)


def _member_response(member: FamilyMember) -> FamilyMemberResponse:
    return FamilyMemberResponse.model_validate(member)


def _invite_response(invite: FamilyInvite, invite_code: str | None = None) -> FamilyInviteResponse:
    response = FamilyInviteResponse.model_validate(invite)
    response.invite_code = invite_code
    return response


def _sent_invite_response(invite: FamilyInvite) -> FamilySentInviteResponse:
    return FamilySentInviteResponse.model_validate(invite)


def _share_setting_response(setting: FamilyShareSetting) -> FamilyShareSettingResponse:
    return FamilyShareSettingResponse.model_validate(setting)


def _notification_setting_response(setting: FamilyNotificationSetting) -> FamilyNotificationSettingResponse:
    return FamilyNotificationSettingResponse.model_validate(setting)


@family_router.post("/groups", response_model=FamilyGroupResponse, status_code=status.HTTP_201_CREATED)
async def create_family_group(
    request: FamilyGroupCreateRequest,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyGroupResponse:
    family = await family_service.create_family_group(user, request)
    return _family_response(family)


@family_router.get("/groups", response_model=list[FamilyGroupResponse])
async def list_family_groups(user: Annotated[User, Depends(get_request_user)]) -> list[FamilyGroupResponse]:
    families = await family_service.list_my_family_groups(user)
    return [_family_response(family) for family in families]


@family_router.get("/groups/{family_id}", response_model=FamilyGroupDetailResponse)
async def get_family_group(
    family_id: int,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyGroupDetailResponse:
    family, members = await family_service.get_family_group_detail(user, family_id)
    return FamilyGroupDetailResponse(
        **_family_response(family).model_dump(),
        members=[_member_response(member) for member in members],
    )


@family_router.patch("/groups/{family_id}", response_model=FamilyGroupResponse)
async def update_family_group(
    family_id: int,
    request: FamilyGroupUpdateRequest,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyGroupResponse:
    family = await family_service.update_family_group(user, family_id, request)
    return _family_response(family)


@family_router.delete("/groups/{family_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_family_group(family_id: int, user: Annotated[User, Depends(get_request_user)]) -> None:
    await family_service.remove_family_group(user, family_id)


@family_router.get("/groups/{family_id}/members", response_model=list[FamilyMemberResponse])
async def list_family_members(
    family_id: int,
    user: Annotated[User, Depends(get_request_user)],
) -> list[FamilyMemberResponse]:
    members = await family_service.list_family_members(user, family_id)
    return [_member_response(member) for member in members]


@family_router.post(
    "/groups/{family_id}/members/unregistered",
    response_model=FamilyMemberResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_unregistered_family_member(
    family_id: int,
    request: FamilyMemberCreateUnregisteredRequest,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyMemberResponse:
    member = await family_service.add_unregistered_family_member(user, family_id, request)
    return _member_response(member)


@family_router.delete("/members/{member_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_family_member(member_id: int, user: Annotated[User, Depends(get_request_user)]) -> None:
    await family_service.remove_family_member(user, member_id)


@family_router.post(
    "/groups/{family_id}/invites",
    response_model=FamilyInviteResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_family_invite(
    family_id: int,
    request: FamilyInviteCreateRequest,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyInviteResponse:
    invite, invite_code = await family_service.create_family_invite(user, family_id, request)
    return _invite_response(invite, invite_code if _allow_family_invite_debug_response() else None)


@family_router.get("/groups/{family_id}/invites", response_model=list[FamilySentInviteResponse])
async def list_family_invites(
    family_id: int,
    user: Annotated[User, Depends(get_request_user)],
) -> list[FamilySentInviteResponse]:
    invites = await family_service.list_family_invites(user, family_id)
    return [_sent_invite_response(invite) for invite in invites]


@family_router.get("/invites/me", response_model=list[FamilyInviteResponse])
async def list_my_family_invites(user: Annotated[User, Depends(get_request_user)]) -> list[FamilyInviteResponse]:
    invites = await family_service.list_my_family_invites(user)
    return [_invite_response(invite) for invite in invites]


@family_router.post("/invites/code/accept", response_model=FamilyMemberResponse)
async def accept_family_invite_by_code(
    request: FamilyInviteAcceptCodeRequest,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyMemberResponse:
    member = await family_service.accept_family_invite_by_code(user, request.code)
    return _member_response(member)


@family_router.post("/invites/{invite_id}/accept", response_model=FamilyMemberResponse)
async def accept_family_invite(
    invite_id: int,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyMemberResponse:
    member = await family_service.accept_family_invite(user, invite_id)
    return _member_response(member)


@family_router.post("/invites/{invite_id}/decline", response_model=FamilyInviteResponse)
async def decline_family_invite(
    invite_id: int,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyInviteResponse:
    invite = await family_service.decline_family_invite(user, invite_id)
    return _invite_response(invite)


@family_router.get("/share-settings", response_model=list[FamilyShareSettingResponse])
async def list_my_family_share_settings(
    user: Annotated[User, Depends(get_request_user)],
) -> list[FamilyShareSettingResponse]:
    settings = await family_service.list_family_share_settings(user)
    return [_share_setting_response(setting) for setting in settings]


@family_router.get("/groups/{family_id}/share-settings", response_model=list[FamilyShareSettingResponse])
async def list_family_group_share_settings(
    family_id: int,
    user: Annotated[User, Depends(get_request_user)],
) -> list[FamilyShareSettingResponse]:
    settings = await family_service.list_family_share_settings(user, family_id=family_id)
    return [_share_setting_response(setting) for setting in settings]


@family_router.patch("/share-settings/{setting_id}", response_model=FamilyShareSettingResponse)
async def update_family_share_setting(
    setting_id: int,
    request: FamilyShareSettingUpdateRequest,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyShareSettingResponse:
    setting = await family_service.update_family_share_setting(user, setting_id, request)
    return _share_setting_response(setting)


@family_router.get("/{family_user_id}/share-settings", response_model=FamilyShareSettingResponse)
async def get_family_action_share_setting(
    family_user_id: int,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyShareSettingResponse:
    setting = await family_service.get_family_action_share_setting(user, family_user_id)
    return _share_setting_response(setting)


@family_router.put("/{family_user_id}/share-settings", response_model=FamilyShareSettingResponse)
async def update_family_action_share_setting(
    family_user_id: int,
    request: FamilyActionShareSettingUpdateRequest,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyShareSettingResponse:
    setting = await family_service.update_family_action_share_setting(user, family_user_id, request)
    return _share_setting_response(setting)


@family_router.get("/{owner_user_id}/notification-settings", response_model=FamilyNotificationSettingResponse)
async def get_family_notification_setting(
    owner_user_id: int,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyNotificationSettingResponse:
    setting = await family_service.get_family_notification_setting(user, owner_user_id)
    return _notification_setting_response(setting)


@family_router.put("/{owner_user_id}/notification-settings", response_model=FamilyNotificationSettingResponse)
async def update_family_notification_setting(
    owner_user_id: int,
    request: FamilyNotificationSettingUpdateRequest,
    user: Annotated[User, Depends(get_request_user)],
) -> FamilyNotificationSettingResponse:
    setting = await family_service.update_family_notification_setting(user, owner_user_id, request)
    return _notification_setting_response(setting)
