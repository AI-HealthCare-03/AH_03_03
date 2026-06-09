from __future__ import annotations

from ai_runtime.cv.providers.gpt_vision import AnalysisType, VisionClient
from ai_runtime.ocr.medication import MedicationOcrItem, parse_medication_text
from app.core import config
from app.dtos.medications import MedicationOCRItem, MedicationOCRRequest, MedicationOCRResponse
from app.services import medications as medication_service

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


async def run_medication_ocr_from_job(job_id: int) -> MedicationOCRResponse:
    from app.services import async_jobs as async_job_service

    job = await async_job_service.get_job(job_id)
    if job is None:
        raise ValueError("medication_ocr_job_not_found")

    payload = job.request_payload or {}
    request = MedicationOCRRequest(
        source_type=str(payload.get("source_type") or "PRESCRIPTION"),
        image_filename=str(payload.get("image_filename") or "") or None,
        memo=None,
        raw_text=medication_service._read_medication_ocr_text(payload.get("text_path")),
    )
    image_bytes = medication_service._read_medication_ocr_bytes(payload.get("upload_path"))
    if image_bytes is None and request.raw_text is None:
        raise ValueError("medication_ocr_source_missing")

    image_media_type = str(payload.get("image_media_type") or "") or medication_service._media_type_from_upload_path(
        str(payload.get("upload_path") or "")
    )
    return await run_medication_ocr(
        request,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
    )


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
