from app.dtos.diets import (
    DietDummyAnalyzeRequest,
    DietPhotoResultCreateRequest,
    DietRecordCreateRequest,
    DietRecordUpdateRequest,
)
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


async def run_dummy_diet_analysis(user_id: int, request: DietDummyAnalyzeRequest) -> dict[str, object]:
    detected_foods = [
        {"name": "현미밥", "confidence": 0.92},
        {"name": "닭가슴살", "confidence": 0.88},
        {"name": "샐러드", "confidence": 0.84},
    ]
    nutrition_summary = {
        "calories": 620,
        "carbohydrate_g": 72,
        "protein_g": 38,
        "fat_g": 18,
        "sodium_mg": 780,
    }
    diet_score = 82.5
    diet_feedback = "단백질과 채소 구성이 좋은 편입니다. 나트륨은 조금만 더 낮춰보세요."
    diet_record = await create_diet_record(
        user_id,
        DietRecordCreateRequest(
            meal_type=request.meal_type,
            meal_time=request.meal_time,
            description=request.description or "더미 식단 분석 기록",
            image_path=request.image_path,
            detected_foods=detected_foods,
            nutrition_summary=nutrition_summary,
            diet_score=diet_score,
            diet_feedback=diet_feedback,
            analysis_method="DUMMY_CV",
            memo=request.memo,
        ),
    )
    photo_result = await create_diet_photo_result(
        diet_record.id,
        DietPhotoResultCreateRequest(
            detected_foods=detected_foods,
            confidence_payload={"method": "dummy", "average_confidence": 0.88},
            raw_output={"source": "mvp_dummy_analyze", "foods": detected_foods},
            is_dummy=True,
        ),
    )
    return {
        "message": "더미 식단 분석이 완료되었습니다.",
        "diet_record": diet_record,
        "photo_result": photo_result,
        "detected_foods": detected_foods,
        "nutrition_summary": nutrition_summary,
        "diet_score": diet_score,
        "diet_feedback": diet_feedback,
    }
