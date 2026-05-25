from ai_runtime.ocr.medication.parser import parse_medication_text
from app.dtos.medications import MedicationOCRRequest
from app.services import medications as medication_service


def test_parse_medication_text_extracts_core_fields() -> None:
    raw_text = """
    안심약국
    처방일자: 2026-05-20
    아세트정 500mg 하루 3회 5일 식후 복용
    """

    result = parse_medication_text(raw_text)

    assert result.pharmacy_name == "안심약국"
    assert result.prescribed_date is not None
    assert len(result.items) == 1
    item = result.items[0]
    assert item.medication_name == "아세트정"
    assert item.dosage == "500mg"
    assert item.frequency == "하루 3회"
    assert item.duration_days == 5
    assert "식후" in (item.instruction or "")


def test_parse_medication_text_handles_number_and_unit_variants() -> None:
    raw_text = "테스트캡슐 0.5 g 1일 2회 14일분 아침 저녁 복용"

    result = parse_medication_text(raw_text)

    assert len(result.items) == 1
    item = result.items[0]
    assert item.medication_name == "테스트캡슐"
    assert item.dosage == "0.5g"
    assert item.frequency == "하루 2회"
    assert item.duration_days == 14
    assert "아침" in (item.instruction or "")
    assert "저녁" in (item.instruction or "")


def test_parse_medication_text_returns_empty_result_safely() -> None:
    result = parse_medication_text("")

    assert result.items == []
    assert result.raw_text == ""
    assert result.warnings == ["empty_raw_text"]


def test_parse_medication_text_returns_warning_when_no_item_detected() -> None:
    result = parse_medication_text("테스트약국\n처방일자 2026-05-20")

    assert result.items == []
    assert result.warnings == ["no_medication_item_detected"]


async def test_medication_ocr_uses_gpt_vision_provider_when_enabled(monkeypatch) -> None:
    class FakeVisionClient:
        def __init__(self, api_key: str, model: str):
            assert api_key == "test-key"
            assert model == "gpt-4o-mini"

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            assert analysis_type == "prescription"
            assert image_bytes == b"image"
            assert media_type == "image/png"
            return {
                "analysis_status": "success",
                "medications": [
                    {
                        "drug_name": "테스트정",
                        "dosage": "10mg",
                        "confidence": 0.87,
                        "raw_text": "테스트정 10mg",
                    }
                ],
            }

    monkeypatch.setattr(medication_service, "VisionClient", FakeVisionClient)
    monkeypatch.setattr(medication_service.config, "MEDICATION_OCR_PROVIDER", "gpt_vision")
    monkeypatch.setattr(medication_service.config, "MEDICATION_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(medication_service.config, "OPENAI_API_KEY", "test-key")

    response = await medication_service.run_medication_ocr(
        MedicationOCRRequest(source_type="PRESCRIPTION"),
        image_bytes=b"image",
        image_media_type="image/png",
    )

    assert response.source == "gpt_vision_medication_ocr"
    assert response.fallback_used is False
    assert response.items[0].name == "테스트정"


async def test_medication_ocr_marks_fallback_when_provider_key_missing(monkeypatch) -> None:
    monkeypatch.setattr(medication_service.config, "MEDICATION_OCR_PROVIDER", "gpt_vision")
    monkeypatch.setattr(medication_service.config, "MEDICATION_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(medication_service.config, "OPENAI_API_KEY", None)

    response = await medication_service.run_medication_ocr(
        MedicationOCRRequest(source_type="PRESCRIPTION"),
        image_bytes=b"image",
        image_media_type="image/png",
    )

    assert response.source == "fallback_medication_ocr"
    assert response.fallback_used is True
    assert response.items
