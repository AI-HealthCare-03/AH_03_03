from typing import Annotated

from fastapi import APIRouter, Depends

from app.apis.v1.dependencies import get_request_user_with_firebase
from app.dtos.main import MainPublicResponse, MainSummaryResponse
from app.models.users import User
from app.services import main as main_service

main_router = APIRouter(prefix="/main", tags=["main"])


@main_router.get("/public", response_model=MainPublicResponse)
async def get_public_main():
    return await main_service.get_public_main()


@main_router.get("/summary", response_model=MainSummaryResponse)
async def get_login_main_summary(user: Annotated[User, Depends(get_request_user_with_firebase)]):
    return await main_service.get_login_main_summary(user)
