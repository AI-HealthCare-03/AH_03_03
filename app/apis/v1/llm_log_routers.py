from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.dependencies.security import get_request_user
from app.dtos.llm_logs import LLMGenerationLogCreateRequest, LLMGenerationLogResponse
from app.models.users import User
from app.services import llm_logs as llm_log_service

llm_log_router = APIRouter(prefix="/llm/logs", tags=["llm_logs"])


@llm_log_router.post("", response_model=LLMGenerationLogResponse, status_code=status.HTTP_201_CREATED)
async def create_llm_generation_log(
    request: LLMGenerationLogCreateRequest, user: Annotated[User, Depends(get_request_user)]
):
    return await llm_log_service.create_llm_generation_log(user.id, request)


@llm_log_router.get("", response_model=list[LLMGenerationLogResponse])
async def list_llm_generation_logs(
    user: Annotated[User, Depends(get_request_user)],
    target_type: str | None = None,
    target_id: int | None = None,
    llm_task_type: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    return await llm_log_service.list_llm_generation_logs(
        user_id=user.id,
        target_type=target_type,
        target_id=target_id,
        llm_task_type=llm_task_type,
        status=status,
        limit=limit,
        offset=offset,
    )


@llm_log_router.get("/{log_id}", response_model=LLMGenerationLogResponse | None)
async def get_llm_generation_log(log_id: int):
    return await llm_log_service.get_llm_generation_log(log_id)
