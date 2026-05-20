from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import ORJSONResponse as Response

from app.apis.v1.dependencies import get_request_user_with_firebase
from app.dtos.users import UserInfoResponse, UserUpdateRequest
from app.models.users import User
from app.services.users import UserManageService

user_router = APIRouter(prefix="/users", tags=["users"])


@user_router.get("/me", response_model=UserInfoResponse, status_code=status.HTTP_200_OK)
async def user_me_info(
    user: Annotated[User, Depends(get_request_user_with_firebase)],
) -> Response:
    return Response(UserInfoResponse.model_validate(user).model_dump(), status_code=status.HTTP_200_OK)


@user_router.patch("/me", response_model=UserInfoResponse, status_code=status.HTTP_200_OK)
async def update_user_me_info(
    update_data: UserUpdateRequest,
    user: Annotated[User, Depends(get_request_user_with_firebase)],
    user_manage_service: Annotated[UserManageService, Depends(UserManageService)],
) -> Response:
    updated_user = await user_manage_service.update_user(user=user, data=update_data)
    return Response(UserInfoResponse.model_validate(updated_user).model_dump(), status_code=status.HTTP_200_OK)


@user_router.delete("/me", response_model=UserInfoResponse, status_code=status.HTTP_200_OK)
async def deactivate_user_me(
    user: Annotated[User, Depends(get_request_user_with_firebase)],
    user_manage_service: Annotated[UserManageService, Depends(UserManageService)],
) -> Response:
    deactivated_user = await user_manage_service.deactivate_user(user)
    return Response(UserInfoResponse.model_validate(deactivated_user).model_dump(), status_code=status.HTTP_200_OK)
