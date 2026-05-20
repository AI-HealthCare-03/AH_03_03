from typing import Any

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
    case = _select_dummy_diet_case(request)
    diet_record = await create_diet_record(
        user_id,
        DietRecordCreateRequest(
            meal_type=request.meal_type,
            meal_time=request.meal_time,
            description=request.description or "더미 식단 분석 기록",
            image_path=request.image_path,
            detected_foods=case["detected_foods"],
            nutrition_summary=case["nutrition_summary"],
            diet_score=case["diet_score"],
            diet_feedback=case["diet_feedback"],
            analysis_method="DUMMY_CV",
            memo=request.memo,
        ),
    )
    photo_result = await create_diet_photo_result(
        diet_record.id,
        DietPhotoResultCreateRequest(
            detected_foods=case["detected_foods"],
            confidence_payload={"method": "dummy", "average_confidence": case["average_confidence"]},
            raw_output={"source": "mvp_dummy_analyze", "case": case["case_name"], "foods": case["detected_foods"]},
            is_dummy=True,
        ),
    )
    return {
        "message": "더미 식단 분석이 완료되었습니다.",
        "diet_record": diet_record,
        "photo_result": photo_result,
        "detected_foods": case["detected_foods"],
        "nutrition_summary": case["nutrition_summary"],
        "diet_score": case["diet_score"],
        "diet_feedback": case["diet_feedback"],
        "warnings": case["warnings"],
        "recommended_actions": case["recommended_actions"],
    }


def _select_dummy_diet_case(request: DietDummyAnalyzeRequest) -> dict[str, Any]:
    text = f"{request.meal_type or ''} {request.description or ''}".lower()
    if "breakfast" in text or "아침" in text:
        return _diet_case(
            "breakfast",
            [
                {"name": "오트밀", "confidence": 0.91},
                {"name": "삶은 달걀", "confidence": 0.88},
                {"name": "블루베리", "confidence": 0.82},
            ],
            {"calories": 430, "carbohydrate_g": 48, "protein_g": 22, "fat_g": 14, "sodium_mg": 420},
            88.0,
            "아침 식사로 균형이 좋습니다. 단백질 구성을 유지해 보세요.",
            [],
            ["물 한 컵 함께 마시기", "오전 간식은 견과류 소량 선택"],
            0.87,
        )
    if "lunch" in text or "점심" in text:
        return _diet_case(
            "lunch",
            [
                {"name": "현미밥", "confidence": 0.92},
                {"name": "닭가슴살", "confidence": 0.88},
                {"name": "샐러드", "confidence": 0.84},
            ],
            {"calories": 620, "carbohydrate_g": 72, "protein_g": 38, "fat_g": 18, "sodium_mg": 780},
            82.5,
            "단백질과 채소 구성이 좋은 편입니다. 나트륨은 조금만 더 낮춰보세요.",
            ["소스류 나트륨 확인 필요"],
            ["드레싱은 절반만 사용", "식후 10분 산책"],
            0.88,
        )
    if "dinner" in text or "저녁" in text:
        return _diet_case(
            "dinner",
            [
                {"name": "잡곡밥", "confidence": 0.86},
                {"name": "된장찌개", "confidence": 0.81},
                {"name": "생선구이", "confidence": 0.87},
            ],
            {"calories": 710, "carbohydrate_g": 80, "protein_g": 34, "fat_g": 24, "sodium_mg": 1280},
            74.0,
            "저녁으로는 나트륨이 다소 높을 수 있습니다. 국물 섭취를 줄여보세요.",
            ["나트륨 주의"],
            ["국물은 절반 이하로 섭취", "야식 피하기"],
            0.85,
        )
    if "snack" in text or "간식" in text:
        return _diet_case(
            "snack",
            [{"name": "카페라떼", "confidence": 0.83}, {"name": "쿠키", "confidence": 0.79}],
            {"calories": 360, "carbohydrate_g": 48, "protein_g": 8, "fat_g": 14, "sodium_mg": 210},
            58.0,
            "당류와 정제 탄수화물이 많은 간식입니다. 대체 간식을 고려해 보세요.",
            ["당류 섭취 주의"],
            ["무가당 음료 선택", "견과류나 요거트로 대체"],
            0.81,
        )
    return _diet_case(
        "default",
        [{"name": "일반식", "confidence": 0.76}, {"name": "채소반찬", "confidence": 0.72}],
        {"calories": 650, "carbohydrate_g": 78, "protein_g": 25, "fat_g": 22, "sodium_mg": 900},
        70.0,
        "식단 정보가 부족해 일반식 기준으로 더미 분석했습니다.",
        ["정확한 음식명 입력 시 더 나은 분석 가능"],
        ["음식 설명 자세히 입력", "채소와 단백질 비율 확인"],
        0.74,
    )


def _diet_case(
    case_name: str,
    detected_foods: list[dict[str, Any]],
    nutrition_summary: dict[str, Any],
    diet_score: float,
    diet_feedback: str,
    warnings: list[str],
    recommended_actions: list[str],
    average_confidence: float,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "detected_foods": detected_foods,
        "nutrition_summary": nutrition_summary,
        "diet_score": diet_score,
        "diet_feedback": diet_feedback,
        "warnings": warnings,
        "recommended_actions": recommended_actions,
        "average_confidence": average_confidence,
    }
