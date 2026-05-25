from fastapi import HTTPException
from starlette import status

from ai_runtime.cv.providers.gpt_vision import AnalysisType, VisionClient
from ai_runtime.ocr.medication import MedicationOcrItem, parse_medication_text
from app.core import config
from app.dtos.medications import (
    MedicationCreateRequest,
    MedicationOCRConfirmRequest,
    MedicationOCRConfirmResponse,
    MedicationOCRItem,
    MedicationOCRRequest,
    MedicationOCRResponse,
    MedicationRecordCreateRequest,
    MedicationRecordUpdateRequest,
    MedicationUpdateRequest,
)
from app.models.medications import Medication, MedicationRecord
from app.repositories import medication_repository

MEDICATION_NOT_FOUND_MESSAGE = "복약/영양제 정보를 찾을 수 없습니다."
MEDICATION_RECORD_NOT_FOUND_MESSAGE = "복약 기록을 찾을 수 없습니다."
MEDICATION_RECORD_ACCESS_DENIED_MESSAGE = "복약 기록에 접근할 수 없습니다."

_FALLBACK_OCR_ITEMS = [
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


async def run_medication_ocr(
    request: MedicationOCRRequest,
    image_bytes: bytes | None = None,
    image_media_type: str | None = None,
) -> MedicationOCRResponse:
    source_type = request.source_type or "PRESCRIPTION"
    provider_result = await _extract_medication_text_with_provider(
        image_bytes=image_bytes,
        image_media_type=image_media_type,
        source_type=source_type,
    )
    raw_text = provider_result["raw_text"] or request.raw_text or request.memo or ""
    parsed = parse_medication_text(raw_text)
    items = _parsed_ocr_items(parsed.items)
    if not items and provider_result["items"]:
        items = provider_result["items"]
    if not items:
        items = _FALLBACK_OCR_ITEMS
        provider_result = {
            **provider_result,
            "source": "fallback_medication_ocr",
            "fallback_used": True,
            "provider_message": provider_result["provider_message"] or "fallback_items_used",
        }
    return MedicationOCRResponse(
        source_type=source_type,
        ocr_confidence=_average_confidence(items),
        items=items,
        message="복약 정보 후보입니다. 실제 처방전/약봉투 내용과 대조한 뒤 저장해주세요.",
        source=provider_result["source"] if provider_result["source"] != "manual_text" else parsed.source,
        fallback_used=provider_result["fallback_used"],
        provider_message=provider_result["provider_message"],
        extracted_text_preview=_preview_text(raw_text),
        raw_text=parsed.raw_text or provider_result["raw_text"] or None,
        parser_warnings=parsed.warnings,
    )


async def _extract_medication_text_with_provider(
    image_bytes: bytes | None,
    image_media_type: str | None,
    source_type: str,
) -> dict[str, object]:
    provider = str(config.MEDICATION_OCR_PROVIDER or "fallback").lower()
    if provider == "gpt_vision" and config.MEDICATION_GPT_VISION_ENABLED:
        result = await _extract_medication_with_gpt_vision(image_bytes, image_media_type, source_type)
        if result is not None:
            return result
        return _base_medication_provider_result("fallback_medication_ocr", True, "gpt_vision_failed_or_unavailable")

    if provider == "paddleocr" and config.PADDLE_OCR_ENABLED:
        result = await _extract_medication_with_paddleocr(image_bytes, image_media_type)
        if result is not None:
            return result
        return _base_medication_provider_result("fallback_medication_ocr", True, "paddleocr_failed_or_unavailable")

    return _base_medication_provider_result("manual_text", not bool(image_bytes), f"{provider}_disabled")


async def _extract_medication_with_gpt_vision(
    image_bytes: bytes | None,
    image_media_type: str | None,
    source_type: str,
) -> dict[str, object] | None:
    if not image_bytes or not config.OPENAI_API_KEY:
        return None
    try:
        client = VisionClient(api_key=config.OPENAI_API_KEY, model=config.MEDICATION_GPT_VISION_MODEL)
        raw = await client.analyze(
            analysis_type=AnalysisType.PRESCRIPTION,
            image_bytes=image_bytes,
            media_type=image_media_type or "image/jpeg",
        )
    except Exception:
        return None

    medications = raw.get("medications") if isinstance(raw, dict) else None
    if not isinstance(medications, list):
        return None
    items = [
        MedicationOCRItem(
            temp_id=f"med-ocr-gpt-{index:03d}",
            name=str(item.get("drug_name") or item.get("name") or "").strip(),
            dosage=str(item.get("dosage") or "") or None,
            frequency=None,
            time_slots=[],
            duration_days=None,
            memo=str(item.get("raw_text") or "") or None,
            confidence=float(item.get("confidence")) if _is_number(item.get("confidence")) else None,
        )
        for index, item in enumerate(medications, start=1)
        if isinstance(item, dict) and str(item.get("drug_name") or item.get("name") or "").strip()
    ]
    raw_text = "\n".join(
        str(item.get("raw_text") or item.get("drug_name") or "") for item in medications if isinstance(item, dict)
    )
    if not items and not raw_text:
        return None
    return {
        "source": "gpt_vision_medication_ocr",
        "fallback_used": False,
        "provider_message": f"gpt_vision_{source_type.lower()}_ocr",
        "raw_text": raw_text,
        "items": items,
    }


async def _extract_medication_with_paddleocr(
    image_bytes: bytes | None,
    image_media_type: str | None,
) -> dict[str, object] | None:
    if not image_bytes:
        return None
    try:
        from ai_runtime.ocr.checkup.extractor import run_ocr, run_ocr_on_pdf
    except Exception:
        return None

    try:
        if image_media_type == "application/pdf":
            _data, _low_confidence_fields, raw_text_lines, _status = await run_ocr_on_pdf(image_bytes)
        else:
            _data, _low_confidence_fields, raw_text_lines, _status = await run_ocr(image_bytes)
    except Exception:
        return None

    raw_text = "\n".join(str(line[0] if isinstance(line, tuple) else line) for line in raw_text_lines)
    if not raw_text.strip():
        return None
    return {
        "source": "paddleocr_medication_ocr",
        "fallback_used": False,
        "provider_message": "paddleocr_medication_ocr",
        "raw_text": raw_text,
        "items": [],
    }


def _base_medication_provider_result(
    source: str, fallback_used: bool, provider_message: str | None
) -> dict[str, object]:
    return {
        "source": source,
        "fallback_used": fallback_used,
        "provider_message": provider_message,
        "raw_text": "",
        "items": [],
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


def _parsed_ocr_items(items: list[MedicationOcrItem]) -> list[MedicationOCRItem]:
    return [
        MedicationOCRItem(
            temp_id=f"med-ocr-{index:03d}",
            name=item.medication_name,
            dosage=item.dosage,
            frequency=item.frequency,
            time_slots=_instruction_time_slots(item.instruction),
            duration_days=item.duration_days,
            memo=item.instruction,
            confidence=item.confidence,
        )
        for index, item in enumerate(items, start=1)
    ]


def _instruction_time_slots(instruction: str | None) -> list[str]:
    if not instruction:
        return []
    slots = []
    for label in ("아침", "점심", "저녁", "취침"):
        if label in instruction:
            slots.append(label)
    return slots


def _average_confidence(items: list[MedicationOCRItem]) -> float:
    scores = [item.confidence for item in items if item.confidence is not None]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


def _preview_text(value: str | None, limit: int = 300) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.split())
    return normalized[:limit] if normalized else None


def _is_number(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


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
