from app.dtos.health import HealthRecordCreateRequest, HealthRecordUpdateRequest
from app.models.health import HealthRecord
from app.repositories import health_repository

REQUIRED_ANALYSIS_FIELDS = {
    "height_cm": "키",
    "weight_kg": "몸무게",
    "bmi": "BMI",
    "fasting_glucose": "공복혈당",
    "ldl_cholesterol": "LDL 콜레스테롤",
    "hdl_cholesterol": "HDL 콜레스테롤",
    "triglyceride": "중성지방",
}


async def create_health_record(user_id: int, request: HealthRecordCreateRequest) -> HealthRecord:
    return await health_repository.create_health_record(user_id, request.model_dump())


async def get_health_record(record_id: int) -> HealthRecord | None:
    return await health_repository.get_health_record_by_id(record_id)


async def get_latest_health_record(user_id: int) -> HealthRecord | None:
    return await health_repository.get_latest_health_record_by_user(user_id)


async def list_health_records(user_id: int, limit: int = 20, offset: int = 0) -> list[HealthRecord]:
    return await health_repository.list_health_records_by_user(user_id, limit=limit, offset=offset)


async def update_health_record(record_id: int, request: HealthRecordUpdateRequest) -> HealthRecord | None:
    data = request.model_dump(exclude_unset=True)
    return await health_repository.update_health_record(record_id, data)


async def delete_health_record(record_id: int) -> int:
    return await health_repository.delete_health_record(record_id)


async def get_analysis_readiness(user_id: int) -> dict[str, object]:
    latest_record = await get_latest_health_record(user_id)
    if latest_record is None:
        return {
            "is_ready": False,
            "latest_health_record_id": None,
            "missing_fields": list(REQUIRED_ANALYSIS_FIELDS.values()),
            "message": "건강 분석을 시작하려면 먼저 건강 정보를 입력해 주세요.",
        }

    missing_fields = [
        label for field_name, label in REQUIRED_ANALYSIS_FIELDS.items() if getattr(latest_record, field_name) is None
    ]
    is_ready = not missing_fields
    message = "건강 분석을 실행할 수 있습니다." if is_ready else "건강 분석에 필요한 항목을 더 입력해 주세요."
    return {
        "is_ready": is_ready,
        "latest_health_record_id": latest_record.id,
        "missing_fields": missing_fields,
        "message": message,
    }
