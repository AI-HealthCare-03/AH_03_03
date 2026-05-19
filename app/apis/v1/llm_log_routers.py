from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import ensure_admin_user, ensure_found, ensure_owner_or_admin, is_admin_user
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
    user_id: int | None = None,
    target_type: str | None = None,
    target_id: int | None = None,
    llm_task_type: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    if user_id is not None and user_id != user.id:
        ensure_admin_user(user)
    effective_user_id = user_id
    if effective_user_id is None and not is_admin_user(user):
        effective_user_id = user.id

    return await llm_log_service.list_llm_generation_logs(
        user_id=effective_user_id,
        target_type=target_type,
        target_id=target_id,
        llm_task_type=llm_task_type,
        status=status,
        limit=limit,
        offset=offset,
    )


@llm_log_router.get("/{log_id}", response_model=LLMGenerationLogResponse)
async def get_llm_generation_log(log_id: int, user: Annotated[User, Depends(get_request_user)]):
    log = ensure_found(await llm_log_service.get_llm_generation_log(log_id), "LLM 생성 로그를 찾을 수 없습니다.")
    ensure_owner_or_admin(log.user_id, user)
    return log
