from typing import Any

from app.models.diets import DietPhotoResult, DietRecord


async def create_diet_record(user_id: int, data: dict[str, Any]) -> DietRecord:
    return await DietRecord.create(user_id=user_id, **data)


async def get_diet_record_by_id(diet_record_id: int) -> DietRecord | None:
    return await DietRecord.get_or_none(id=diet_record_id)


async def list_diet_records_by_user(
    user_id: int, analysis_method: str | None = None, limit: int = 20, offset: int = 0
) -> list[DietRecord]:
    query = DietRecord.filter(user_id=user_id)
    if analysis_method is not None:
        query = query.filter(analysis_method=analysis_method)
    return await query.order_by("-created_at").offset(offset).limit(limit)


async def update_diet_record(diet_record_id: int, data: dict[str, Any]) -> DietRecord | None:
    diet_record = await get_diet_record_by_id(diet_record_id)
    if diet_record is None:
        return None
    for key, value in data.items():
        setattr(diet_record, key, value)
    await diet_record.save(update_fields=list(data.keys()) if data else None)
    return diet_record


async def delete_diet_record(diet_record_id: int) -> int:
    return await DietRecord.filter(id=diet_record_id).delete()


async def create_diet_photo_result(diet_record_id: int, data: dict[str, Any]) -> DietPhotoResult:
    return await DietPhotoResult.create(diet_record_id=diet_record_id, **data)


async def get_diet_photo_result_by_id(photo_result_id: int) -> DietPhotoResult | None:
    return await DietPhotoResult.get_or_none(id=photo_result_id)


async def list_diet_photo_results(diet_record_id: int, limit: int = 20, offset: int = 0) -> list[DietPhotoResult]:
    return (
        await DietPhotoResult.filter(diet_record_id=diet_record_id).order_by("-created_at").offset(offset).limit(limit)
    )


async def delete_diet_photo_result(photo_result_id: int) -> int:
    return await DietPhotoResult.filter(id=photo_result_id).delete()
