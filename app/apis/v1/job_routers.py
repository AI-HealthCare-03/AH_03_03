from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.apis.v1.dependencies import ensure_found, get_request_user, is_monitor_user, require_monitor_user
from app.dtos.async_jobs import AsyncJobResponse, DemoJobCreateRequest
from app.models.async_jobs import AsyncJob
from app.models.users import User
from app.services import async_jobs as async_job_service

job_router = APIRouter(prefix="/jobs", tags=["jobs"])


@job_router.post("/demo", response_model=AsyncJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_demo_async_job(
    request: DemoJobCreateRequest,
    _user: Annotated[User, Depends(require_monitor_user)],
):
    payload = request.model_dump()
    return await async_job_service.create_demo_job(payload)


@job_router.get("/{job_id}", response_model=AsyncJobResponse)
async def get_async_job(
    job_id: int,
    user: Annotated[User, Depends(get_request_user)],
):
    job = ensure_found(await async_job_service.get_job(job_id), "비동기 작업을 찾을 수 없습니다.")
    if not _can_read_job(job, user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="해당 작업에 접근할 권한이 없습니다.")
    return job


def _can_read_job(job: AsyncJob, user: User) -> bool:
    if is_monitor_user(user):
        return True
    job_user_id = _payload_int(job.request_payload or {}, "user_id")
    return job_user_id is not None and job_user_id == int(user.id)


def _payload_int(payload: dict, key: str) -> int | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
