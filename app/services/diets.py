from app.dtos.diets import DietPhotoResultCreateRequest, DietRecordCreateRequest, DietRecordUpdateRequest
from app.models.diets import DietPhotoResult, DietRecord
from app.repositories import diet_repository


async def create_diet_record(user_id: int, request: DietRecordCreateRequest) -> DietRecord:
    return await diet_repository.create_diet_record(user_id, request.model_dump())


async def get_diet_record(diet_record_id: int) -> DietRecord | None:
    return await diet_repository.get_diet_record_by_id(diet_record_id)


async def list_diet_records(
    user_id: int, analysis_method: str | None = None, limit: int = 20, offset: int = 0
) -> list[DietRecord]:
    return await diet_repository.list_diet_records_by_user(
        user_id=user_id,
        analysis_method=analysis_method,
        limit=limit,
        offset=offset,
    )


async def update_diet_record(diet_record_id: int, request: DietRecordUpdateRequest) -> DietRecord | None:
    return await diet_repository.update_diet_record(diet_record_id, request.model_dump(exclude_unset=True))


async def delete_diet_record(diet_record_id: int) -> int:
    return await diet_repository.delete_diet_record(diet_record_id)


async def create_diet_photo_result(diet_record_id: int, request: DietPhotoResultCreateRequest) -> DietPhotoResult:
    return await diet_repository.create_diet_photo_result(diet_record_id, request.model_dump())


async def get_diet_photo_result(photo_result_id: int) -> DietPhotoResult | None:
    return await diet_repository.get_diet_photo_result_by_id(photo_result_id)


async def list_diet_photo_results(diet_record_id: int, limit: int = 20, offset: int = 0) -> list[DietPhotoResult]:
    return await diet_repository.list_diet_photo_results(diet_record_id, limit=limit, offset=offset)
