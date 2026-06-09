from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from starlette import status

from app.core import config
from app.dtos.medications import (
    MedicationCreateRequest,
    MedicationOCRConfirmRequest,
    MedicationOCRConfirmResponse,
    MedicationRecordCreateRequest,
    MedicationRecordUpdateRequest,
    MedicationUpdateRequest,
)
from app.models.medications import Medication, MedicationRecord
from app.repositories import medication_repository
from app.services.storage import get_storage_service, normalize_storage_key

MEDICATION_NOT_FOUND_MESSAGE = "복약/영양제 정보를 찾을 수 없습니다."
MEDICATION_RECORD_NOT_FOUND_MESSAGE = "복약 기록을 찾을 수 없습니다."
MEDICATION_RECORD_ACCESS_DENIED_MESSAGE = "복약 기록에 접근할 수 없습니다."
MEDICATION_OCR_UPLOAD_DIR = "medication_ocr"
MEDICATION_OCR_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}


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


def store_medication_ocr_upload(
    *,
    user_id: int,
    image_bytes: bytes,
    image_media_type: str | None,
    filename: str | None,
) -> dict[str, str]:
    suffix = _safe_file_suffix(filename, image_media_type)
    storage_key = _build_medication_ocr_key(user_id=user_id, suffix=suffix)
    stored_key = get_storage_service().save_bytes(image_bytes, storage_key, content_type=image_media_type)
    return {
        "upload_path": stored_key,
        "image_media_type": image_media_type or _media_type_from_upload_path(stored_key),
        "image_filename": filename or Path(stored_key).name,
    }


def store_medication_ocr_text(*, user_id: int, text: str) -> dict[str, str]:
    storage_key = _build_medication_ocr_key(user_id=user_id, suffix=".txt")
    text_path = get_storage_service().save_bytes(text.encode("utf-8"), storage_key, content_type="text/plain")
    return {
        "text_path": text_path,
    }


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


def _upload_storage_root() -> Path:
    root = Path(config.UPLOAD_STORAGE_DIR)
    if not root.is_absolute():
        root = Path.cwd() / root
    path = root / MEDICATION_OCR_UPLOAD_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_medication_ocr_path(*, user_id: int, suffix: str) -> Path:
    user_dir = _upload_storage_root() / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / f"{uuid4().hex}{suffix}"


def _build_medication_ocr_key(*, user_id: int, suffix: str) -> str:
    return normalize_storage_key(f"medication-ocr/{user_id}/{uuid4().hex}/source{suffix}")


def _safe_file_suffix(filename: str | None, media_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in MEDICATION_OCR_MEDIA_TYPES:
        return suffix
    if media_type == "image/png":
        return ".png"
    if media_type == "image/webp":
        return ".webp"
    if media_type == "application/pdf":
        return ".pdf"
    return ".jpg"


def _read_medication_ocr_bytes(path_value: object) -> bytes | None:
    if not path_value:
        return None
    try:
        storage = get_storage_service()
        if storage.exists(str(path_value)):
            return storage.read_bytes(str(path_value))
    except Exception:
        pass

    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        raise ValueError("medication_ocr_upload_missing")
    return path.read_bytes()


def _read_medication_ocr_text(path_value: object) -> str | None:
    if not path_value:
        return None
    try:
        storage = get_storage_service()
        if storage.exists(str(path_value)):
            text = storage.read_bytes(str(path_value)).decode("utf-8").strip()
            return text or None
    except Exception:
        pass

    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        raise ValueError("medication_ocr_text_missing")
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _media_type_from_upload_path(path_value: str) -> str | None:
    if not path_value:
        return None
    return MEDICATION_OCR_MEDIA_TYPES.get(Path(path_value).suffix.lower())


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
