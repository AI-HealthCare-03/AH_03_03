from datetime import time

from fastapi import HTTPException
from starlette import status

from app.dtos.medications import (
    MedicationCreateRequest,
    MedicationRecordCreateRequest,
    MedicationRecordUpdateRequest,
    MedicationUpdateRequest,
)
from app.models.medications import Medication, MedicationRecord
from app.repositories import medication_repository

MEDICATION_NOT_FOUND_MESSAGE = "복약/영양제 정보를 찾을 수 없습니다."
MEDICATION_RECORD_NOT_FOUND_MESSAGE = "복약 기록을 찾을 수 없습니다."
MEDICATION_RECORD_ACCESS_DENIED_MESSAGE = "복약 기록에 접근할 수 없습니다."


def _medication_payload(data: dict) -> dict:
    reminder_time = data.get("reminder_time")
    if isinstance(reminder_time, time):
        data["reminder_time"] = reminder_time.replace(tzinfo=None)
    return data


async def create_medication(user_id: int, request: MedicationCreateRequest) -> Medication:
    return await medication_repository.create_medication(user_id, _medication_payload(request.model_dump()))


async def get_medication(medication_id: int) -> Medication | None:
    return await medication_repository.get_medication_by_id(medication_id)


async def list_medications(
    user_id: int,
    is_active: bool | None = None,
    medication_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[Medication]:
    return await medication_repository.list_medications_by_user(
        user_id=user_id,
        is_active=is_active,
        medication_type=medication_type,
        limit=limit,
        offset=offset,
    )


async def update_medication(medication_id: int, request: MedicationUpdateRequest) -> Medication | None:
    return await medication_repository.update_medication(
        medication_id,
        _medication_payload(request.model_dump(exclude_unset=True)),
    )


async def deactivate_medication(medication_id: int) -> Medication | None:
    return await medication_repository.update_medication(medication_id, {"is_active": False})


async def delete_medication(medication_id: int) -> int:
    return await medication_repository.delete_medication(medication_id)


async def create_medication_record(
    medication_id: int, user_id: int, request: MedicationRecordCreateRequest
) -> MedicationRecord:
    medication = await _get_owned_medication_or_raise(medication_id, user_id)
    return await medication_repository.create_medication_record(medication.id, user_id, request.model_dump())


async def get_medication_record(record_id: int, user_id: int | None = None) -> MedicationRecord | None:
    record = await medication_repository.get_medication_record_by_id(record_id)
    if record is not None and user_id is not None:
        await _ensure_record_consistency_or_raise(record, user_id)
    return record


async def list_medication_records(
    user_id: int | None = None,
    medication_id: int | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[MedicationRecord]:
    if user_id is not None and medication_id is not None:
        await _get_owned_medication_or_raise(medication_id, user_id)

    records = await medication_repository.list_medication_records(
        user_id=user_id,
        medication_id=medication_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    if user_id is not None:
        for record in records:
            await _ensure_record_consistency_or_raise(record, user_id)
    return records


async def update_medication_record(
    record_id: int, user_id: int, request: MedicationRecordUpdateRequest
) -> MedicationRecord | None:
    record = await get_medication_record(record_id, user_id=user_id)
    if record is None:
        return None
    return await medication_repository.update_medication_record(record_id, request.model_dump(exclude_unset=True))


async def delete_medication_record(record_id: int, user_id: int) -> int:
    record = await get_medication_record(record_id, user_id=user_id)
    if record is None:
        return 0
    return await medication_repository.delete_medication_record(record_id)


async def _get_owned_medication_or_raise(medication_id: int, user_id: int) -> Medication:
    medication = await get_medication(medication_id)
    if medication is None or int(medication.user_id) != int(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=MEDICATION_NOT_FOUND_MESSAGE)
    return medication


async def _ensure_record_consistency_or_raise(record: MedicationRecord, user_id: int) -> None:
    if int(record.user_id) != int(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=MEDICATION_RECORD_NOT_FOUND_MESSAGE)
    await record.fetch_related("medication")
    if int(record.medication.user_id) != int(record.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=MEDICATION_RECORD_ACCESS_DENIED_MESSAGE,
        )
