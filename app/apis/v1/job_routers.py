from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import ensure_found, require_monitor_user
from app.dtos.async_jobs import AsyncJobResponse, DemoJobCreateRequest
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
    _user: Annotated[User, Depends(require_monitor_user)],
):
    return ensure_found(await async_job_service.get_job(job_id), "비동기 작업을 찾을 수 없습니다.")
