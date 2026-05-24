from fastapi import APIRouter, status

from app.apis.v1.dependencies import ensure_found
from app.dtos.async_jobs import AsyncJobResponse, DemoJobCreateRequest
from app.services import async_jobs as async_job_service

job_router = APIRouter(prefix="/jobs", tags=["jobs"])


@job_router.post("/demo", response_model=AsyncJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_demo_async_job(request: DemoJobCreateRequest):
    payload = request.model_dump()
    return await async_job_service.create_demo_job(payload)


@job_router.get("/{job_id}", response_model=AsyncJobResponse)
async def get_async_job(job_id: int):
    return ensure_found(await async_job_service.get_job(job_id), "비동기 작업을 찾을 수 없습니다.")
