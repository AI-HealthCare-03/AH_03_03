from app.dtos.medications import (
    MedicationCreateRequest,
    MedicationRecordCreateRequest,
    MedicationRecordUpdateRequest,
    MedicationUpdateRequest,
)
from app.models.medications import Medication, MedicationRecord
from app.repositories import medication_repository


async def create_medication(user_id: int, request: MedicationCreateRequest) -> Medication:
    return await medication_repository.create_medication(user_id, request.model_dump())


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
    return await medication_repository.update_medication(medication_id, request.model_dump(exclude_unset=True))


async def deactivate_medication(medication_id: int) -> Medication | None:
    return await medication_repository.update_medication(medication_id, {"is_active": False})


async def create_medication_record(
    medication_id: int, user_id: int, request: MedicationRecordCreateRequest
) -> MedicationRecord:
    return await medication_repository.create_medication_record(medication_id, user_id, request.model_dump())


async def get_medication_record(record_id: int) -> MedicationRecord | None:
    return await medication_repository.get_medication_record_by_id(record_id)


async def list_medication_records(
    user_id: int | None = None,
    medication_id: int | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[MedicationRecord]:
    return await medication_repository.list_medication_records(
        user_id=user_id,
        medication_id=medication_id,
        status=status,
        limit=limit,
        offset=offset,
    )


async def update_medication_record(record_id: int, request: MedicationRecordUpdateRequest) -> MedicationRecord | None:
    return await medication_repository.update_medication_record(record_id, request.model_dump(exclude_unset=True))
