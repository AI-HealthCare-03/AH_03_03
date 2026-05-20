from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import ensure_found, ensure_owner, get_request_user_with_firebase
from app.dtos.diets import (
    DietDummyAnalyzeRequest,
    DietDummyAnalyzeResponse,
    DietPhotoResultCreateRequest,
    DietPhotoResultResponse,
    DietRecordCreateRequest,
    DietRecordResponse,
    DietRecordUpdateRequest,
)
from app.models.users import User
from app.services import diets as diet_service

diet_router = APIRouter(prefix="/diets", tags=["diets"])


@diet_router.post("", response_model=DietRecordResponse, status_code=status.HTTP_201_CREATED)
async def create_diet_record(
    request: DietRecordCreateRequest, user: Annotated[User, Depends(get_request_user_with_firebase)]
):
    return await diet_service.create_diet_record(user.id, request)


@diet_router.get("", response_model=list[DietRecordResponse])
async def list_diet_records(
    user: Annotated[User, Depends(get_request_user_with_firebase)],
    analysis_method: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    return await diet_service.list_diet_records(
        user.id,
        analysis_method=analysis_method,
        limit=limit,
        offset=offset,
    )


@diet_router.post("/dummy-analyze", response_model=DietDummyAnalyzeResponse, status_code=status.HTTP_201_CREATED)
async def run_dummy_diet_analysis(
    request: DietDummyAnalyzeRequest,
    user: Annotated[User, Depends(get_request_user_with_firebase)],
):
    return await diet_service.run_dummy_diet_analysis(user.id, request)


@diet_router.get("/{diet_record_id}", response_model=DietRecordResponse)
async def get_diet_record(diet_record_id: int, user: Annotated[User, Depends(get_request_user_with_firebase)]):
    record = ensure_found(await diet_service.get_diet_record(diet_record_id), "식단 기록을 찾을 수 없습니다.")
    ensure_owner(record.user_id, user)
    return record


@diet_router.patch("/{diet_record_id}", response_model=DietRecordResponse)
async def update_diet_record(
    diet_record_id: int,
    request: DietRecordUpdateRequest,
    user: Annotated[User, Depends(get_request_user_with_firebase)],
):
    record = ensure_found(await diet_service.get_diet_record(diet_record_id), "식단 기록을 찾을 수 없습니다.")
    ensure_owner(record.user_id, user)
    updated = await diet_service.update_diet_record(diet_record_id, request)
    return ensure_found(updated, "식단 기록을 찾을 수 없습니다.")


@diet_router.delete("/{diet_record_id}")
async def delete_diet_record(diet_record_id: int, user: Annotated[User, Depends(get_request_user_with_firebase)]):
    record = ensure_found(await diet_service.get_diet_record(diet_record_id), "식단 기록을 찾을 수 없습니다.")
    ensure_owner(record.user_id, user)
    deleted_count = await diet_service.delete_diet_record(diet_record_id)
    return {"deleted_count": deleted_count}


@diet_router.post(
    "/{diet_record_id}/photo-result", response_model=DietPhotoResultResponse, status_code=status.HTTP_201_CREATED
)
async def create_diet_photo_result(
    diet_record_id: int,
    request: DietPhotoResultCreateRequest,
    user: Annotated[User, Depends(get_request_user_with_firebase)],
):
    record = ensure_found(await diet_service.get_diet_record(diet_record_id), "식단 기록을 찾을 수 없습니다.")
    ensure_owner(record.user_id, user)
    return await diet_service.create_diet_photo_result(diet_record_id, request)


@diet_router.get("/{diet_record_id}/photo-result", response_model=list[DietPhotoResultResponse])
async def list_diet_photo_results(
    diet_record_id: int,
    user: Annotated[User, Depends(get_request_user_with_firebase)],
    limit: int = 20,
    offset: int = 0,
):
    record = ensure_found(await diet_service.get_diet_record(diet_record_id), "식단 기록을 찾을 수 없습니다.")
    ensure_owner(record.user_id, user)
    return await diet_service.list_diet_photo_results(diet_record_id, limit=limit, offset=offset)
