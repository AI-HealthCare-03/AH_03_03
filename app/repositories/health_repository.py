from typing import Any

from app.models.health import HealthRecord


async def create_health_record(user_id: int, data: dict[str, Any]) -> HealthRecord:
    return await HealthRecord.create(user_id=user_id, **data)


async def get_health_record_by_id(record_id: int) -> HealthRecord | None:
    return await HealthRecord.get_or_none(id=record_id)


async def get_latest_health_record_by_user(user_id: int) -> HealthRecord | None:
    return await HealthRecord.filter(user_id=user_id).order_by("-created_at", "-id").first()


async def list_health_records_by_user(user_id: int, limit: int = 20, offset: int = 0) -> list[HealthRecord]:
    return await HealthRecord.filter(user_id=user_id).order_by("-created_at", "-id").offset(offset).limit(limit)


async def update_health_record(record_id: int, data: dict[str, Any]) -> HealthRecord | None:
    record = await get_health_record_by_id(record_id)
    if record is None:
        return None

    for key, value in data.items():
        setattr(record, key, value)
    await record.save(update_fields=list(data.keys()) if data else None)
    return record


async def delete_health_record(record_id: int) -> int:
    return await HealthRecord.filter(id=record_id).delete()
