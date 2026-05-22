from fastapi import HTTPException
from starlette import status

from app.dtos.medications import (
    MedicationCreateRequest,
    MedicationOCRConfirmRequest,
    MedicationOCRConfirmResponse,
    MedicationOCRDummyRequest,
    MedicationOCRDummyResponse,
    MedicationOCRItem,
    MedicationRecordCreateRequest,
    MedicationRecordUpdateRequest,
    MedicationUpdateRequest,
)
from app.models.medications import Medication, MedicationRecord
from app.repositories import medication_repository

MEDICATION_NOT_FOUND_MESSAGE = "복약/영양제 정보를 찾을 수 없습니다."
MEDICATION_RECORD_NOT_FOUND_MESSAGE = "복약 기록을 찾을 수 없습니다."
MEDICATION_RECORD_ACCESS_DENIED_MESSAGE = "복약 기록에 접근할 수 없습니다."

_DUMMY_OCR_ITEMS = [
    MedicationOCRItem(
        temp_id="med-ocr-001",
        name="메트포르민",
        dosage="500mg",
        frequency="하루 2회",
        time_slots=["아침", "저녁"],
        duration_days=30,
        memo="식후 복용",
        confidence=0.96,
    ),
    MedicationOCRItem(
        temp_id="med-ocr-002",
        name="혈압약",
        dosage="5mg",
        frequency="하루 1회",
        time_slots=["아침"],
        duration_days=30,
        memo="매일 같은 시간에 복용",
        confidence=0.93,
    ),
    MedicationOCRItem(
        temp_id="med-ocr-003",
        name="오메가3",
        dosage="1000mg",
        frequency="하루 1회",
        time_slots=["저녁"],
        duration_days=60,
        memo="식후 복용",
        confidence=0.91,
    ),
]


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


async def delete_medication(medication_id: int) -> int:
    return await medication_repository.delete_medication(medication_id)


async def run_dummy_medication_ocr(request: MedicationOCRDummyRequest) -> MedicationOCRDummyResponse:
    source_type = request.source_type or "PRESCRIPTION"
    return MedicationOCRDummyResponse(
        source_type=source_type,
        ocr_confidence=0.93,
        items=_DUMMY_OCR_ITEMS,
        message="MVP 더미 OCR 결과입니다. 실제 OCR/CLOVA 연동은 후속 작업에서 구현합니다.",
    )


async def confirm_medication_ocr(user_id: int, request: MedicationOCRConfirmRequest) -> MedicationOCRConfirmResponse:
    created_medication_ids: list[int] = []
    skipped_count = 0

    for item in request.items:
        name = item.name.strip()
        if not name:
            skipped_count += 1
            continue

        created = await create_medication(
            user_id,
            MedicationCreateRequest(
                name=name,
                medication_type="MEDICATION",
                dosage=item.dosage,
                frequency=item.frequency,
                reminder_time=None,
                is_active=True,
                memo=_build_ocr_medication_memo(
                    duration_days=item.duration_days,
                    time_slots=item.time_slots,
                    memo=item.memo,
                ),
            ),
        )
        created_medication_ids.append(created.id)

    return MedicationOCRConfirmResponse(
        created_count=len(created_medication_ids),
        created_medication_ids=created_medication_ids,
        skipped_count=skipped_count,
        message="확인된 OCR 후보를 복약/영양제 정보로 저장했습니다.",
    )


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


def _build_ocr_medication_memo(duration_days: int | None, time_slots: list[str], memo: str | None) -> str | None:
    memo_parts: list[str] = []
    if duration_days is not None:
        memo_parts.append(f"{duration_days}일")
    if time_slots:
        memo_parts.append(f"복용 시간: {', '.join(time_slots)}")
    if memo:
        memo_parts.append(memo)
    return " / ".join(memo_parts) if memo_parts else None


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
