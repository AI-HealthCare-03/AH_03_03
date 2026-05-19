from app.dtos.health import HealthRecordCreateRequest, HealthRecordUpdateRequest
from app.models.health import HealthRecord
from app.repositories import health_repository


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
