from typing import Any

from app.models.medications import Medication, MedicationRecord


async def create_medication(user_id: int, data: dict[str, Any]) -> Medication:
    return await Medication.create(user_id=user_id, **data)


async def get_medication_by_id(medication_id: int) -> Medication | None:
    return await Medication.get_or_none(id=medication_id)


async def list_medications_by_user(
    user_id: int,
    is_active: bool | None = None,
    medication_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[Medication]:
    query = Medication.filter(user_id=user_id)
    if is_active is not None:
        query = query.filter(is_active=is_active)
    if medication_type is not None:
        query = query.filter(medication_type=medication_type)
    return await query.order_by("-created_at").offset(offset).limit(limit)


async def update_medication(medication_id: int, data: dict[str, Any]) -> Medication | None:
    medication = await get_medication_by_id(medication_id)
    if medication is None:
        return None
    for key, value in data.items():
        setattr(medication, key, value)
    await medication.save(update_fields=list(data.keys()) if data else None)
    return medication


async def delete_medication(medication_id: int) -> int:
    return await Medication.filter(id=medication_id).delete()


async def create_medication_record(medication_id: int, user_id: int, data: dict[str, Any]) -> MedicationRecord:
    return await MedicationRecord.create(medication_id=medication_id, user_id=user_id, **data)


async def get_medication_record_by_id(record_id: int) -> MedicationRecord | None:
    return await MedicationRecord.get_or_none(id=record_id)


async def list_medication_records(
    user_id: int | None = None,
    medication_id: int | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[MedicationRecord]:
    query = MedicationRecord.all()
    if user_id is not None:
        query = query.filter(user_id=user_id)
    if medication_id is not None:
        query = query.filter(medication_id=medication_id)
    if status is not None:
        query = query.filter(status=status)
    return await query.order_by("-created_at").offset(offset).limit(limit)


async def update_medication_record(record_id: int, data: dict[str, Any]) -> MedicationRecord | None:
    record = await get_medication_record_by_id(record_id)
    if record is None:
        return None
    for key, value in data.items():
        setattr(record, key, value)
    await record.save(update_fields=list(data.keys()) if data else None)
    return record


async def delete_medication_record(record_id: int) -> int:
    return await MedicationRecord.filter(id=record_id).delete()
