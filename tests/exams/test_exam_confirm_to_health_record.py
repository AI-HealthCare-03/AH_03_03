from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import exams as exam_service
from app.services.exams import (
    build_health_record_update_from_exam_measurements,
    parse_exam_measurement_number,
)


def test_ldl_and_hdl_keys_map_to_health_record_cholesterol_fields() -> None:
    data = build_health_record_update_from_exam_measurements(
        [
            _measurement("ldl", "130"),
            _measurement("hdl", "46"),
            _measurement("ldl_cholesterol", "128"),
            _measurement("hdl_cholesterol", "48"),
        ]
    )

    assert data["ldl_cholesterol"] == 128
    assert data["hdl_cholesterol"] == 48


def test_numeric_string_with_unit_is_parsed() -> None:
    assert parse_exam_measurement_number("130 mg/dL") == Decimal("130")
    assert parse_exam_measurement_number("5.8 %") == Decimal("5.8")
    assert parse_exam_measurement_number("") is None
    assert parse_exam_measurement_number("음성") is None
    assert parse_exam_measurement_number("비해당") is None
    assert parse_exam_measurement_number("해당 없음") is None
    assert parse_exam_measurement_number("-") is None
    assert parse_exam_measurement_number("없음") is None
    assert parse_exam_measurement_number("미실시") is None
    assert parse_exam_measurement_number("검사안함") is None


def test_non_numeric_exam_values_are_not_measurement_candidates() -> None:
    measurements = exam_service._measurement_tuples_from_mapping(
        {
            "systolic_bp": "비해당",
            "diastolic_bp": "-",
            "fasting_glucose": "검사안함",
            "ldl": "132 mg/dL",
        }
    )

    assert measurements == [("ldl", "LDL 콜레스테롤", "132 mg/dL", "mg/dL")]


def test_numeric_exam_value_takes_priority_over_later_non_numeric_value() -> None:
    merged: dict[str, object] = {}

    exam_service._merge_exam_extracted_data(merged, {"ldl": "67", "hdl": "비해당"})
    exam_service._merge_exam_extracted_data(merged, {"ldl": "비해당", "hdl": "49"})

    assert merged == {"ldl": "67", "hdl": "49"}


def test_exam_pdf_measurement_page_scoring_selects_table_page() -> None:
    page_indices = exam_service._select_exam_measurement_page_indices(
        [
            "종합소견 결과 안내 HDL",
            "검사항목 결과 참고치 공복혈당 총콜레스테롤 HDL LDL 중성지방 AST ALT 혈색소",
            "발급 안내문 결과",
        ]
    )

    assert page_indices == [1]


def test_auto_exam_ocr_provider_order_uses_file_type_policy() -> None:
    assert exam_service._select_exam_ocr_provider_order("auto", "application/pdf", "checkup.pdf") == [
        "paddleocr",
        "gpt_vision",
    ]
    assert exam_service._select_exam_ocr_provider_order("auto", "image/png", "checkup.png") == [
        "gpt_vision",
        "paddleocr",
    ]


def test_exam_measurements_build_health_record_x2_fields() -> None:
    data = build_health_record_update_from_exam_measurements(
        [
            _measurement("systolic_bp", "132 mmHg"),
            _measurement("diastolic_bp", "84 mmHg"),
            _measurement("fasting_glucose", "108 mg/dL"),
            _measurement("total_cholesterol", "210 mg/dL"),
            _measurement("triglyceride", "158 mg/dL"),
            _measurement("hdl", "46 mg/dL"),
            _measurement("ldl", "132 mg/dL"),
            _measurement("hba1c", "5.8 %"),
            _measurement("waist_cm", "89.0 cm"),
        ]
    )

    assert data == {
        "systolic_bp": 132,
        "diastolic_bp": 84,
        "fasting_glucose": 108,
        "total_cholesterol": 210,
        "triglyceride": 158,
        "hdl_cholesterol": 46,
        "ldl_cholesterol": 132,
        "hba1c": Decimal("5.8"),
        "waist_cm": Decimal("89.0"),
    }


def test_height_weight_and_bmi_mapping_recalculates_bmi() -> None:
    data = build_health_record_update_from_exam_measurements(
        [
            _measurement("height", "172.4 cm"),
            _measurement("weight", "76.8 kg"),
            _measurement("bmi", "99.9"),
        ]
    )

    assert data["height_cm"] == Decimal("172.4")
    assert data["weight_kg"] == Decimal("76.8")
    assert data["bmi"] == Decimal("25.84")


def test_invalid_or_none_values_do_not_overwrite_health_record_fields() -> None:
    data = build_health_record_update_from_exam_measurements(
        [
            _measurement("systolic_bp", ""),
            _measurement("fasting_glucose", None),
            _measurement("urine_protein", "음성"),
            _measurement("unknown_key", "100"),
        ]
    )

    assert data == {}


def test_exam_upload_key_uses_user_exam_and_unique_segment() -> None:
    report = SimpleNamespace(id=7, user_id=3)

    key = exam_service._build_exam_upload_key(report, "image/jpeg", "checkup.heic")

    assert key.startswith("exams/3/7/")
    assert key.endswith("/source.jpg")
    assert ".." not in key


def test_exam_upload_media_type_is_inferred_from_stored_path() -> None:
    assert exam_service._media_type_from_upload_path(Path("var/uploads/exams/1/2/source.pdf")) == "application/pdf"
    assert exam_service._media_type_from_upload_path(Path("var/uploads/exams/1/2/source.webp")) == "image/webp"
    assert exam_service._media_type_from_storage_key("exams/1/2/source.pdf") == "application/pdf"


@pytest.mark.asyncio
async def test_store_exam_ocr_upload_uses_local_storage_backend(monkeypatch, tmp_path: Path) -> None:
    report = SimpleNamespace(id=7, user_id=3, original_filename=None)
    captured_update: dict[str, object] = {}
    monkeypatch.setattr(exam_service.config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(exam_service.config, "LOCAL_STORAGE_ROOT", str(tmp_path))

    async def fake_update_exam_report(exam_report_id: int, request):
        assert exam_report_id == 7
        captured_update.update(request.model_dump(exclude_unset=True))
        return SimpleNamespace(**{**report.__dict__, **captured_update})

    monkeypatch.setattr(exam_service, "update_exam_report", fake_update_exam_report)

    updated = await exam_service.store_exam_ocr_upload(
        report,
        b"exam-image",
        "image/jpeg",
        "checkup.jpg",
    )

    stored_key = str(captured_update["file_path"])
    assert stored_key.startswith("exams/3/7/")
    assert stored_key.endswith("/source.jpg")
    assert (tmp_path / stored_key).read_bytes() == b"exam-image"
    assert updated.file_path == stored_key


@pytest.mark.asyncio
async def test_run_exam_ocr_from_report_reads_upload_via_storage(monkeypatch, tmp_path: Path) -> None:
    stored_key = "exams/3/7/test/source.png"
    storage_path = tmp_path / stored_key
    storage_path.parent.mkdir(parents=True)
    storage_path.write_bytes(b"stored-image")
    report = SimpleNamespace(id=7, user_id=3, file_path=stored_key, original_filename="checkup.png")
    captured: dict[str, object] = {}

    monkeypatch.setattr(exam_service.config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(exam_service.config, "LOCAL_STORAGE_ROOT", str(tmp_path))

    async def fake_get_exam_report(exam_report_id: int):
        assert exam_report_id == 7
        return report

    async def fake_update_exam_report(exam_report_id: int, request):
        assert exam_report_id == 7
        return report

    async def fake_run_exam_ocr(**kwargs):
        captured.update(kwargs)
        return "ocr-response"

    monkeypatch.setattr(exam_service, "get_exam_report", fake_get_exam_report)
    monkeypatch.setattr(exam_service, "update_exam_report", fake_update_exam_report)
    monkeypatch.setattr(exam_service, "run_exam_ocr", fake_run_exam_ocr)

    response = await exam_service.run_exam_ocr_from_report(7)

    assert response == "ocr-response"
    assert captured["image_bytes"] == b"stored-image"
    assert captured["image_media_type"] == "image/png"
    assert captured["image_filename"] == "checkup.png"


@pytest.mark.asyncio
async def test_exam_ocr_uses_gpt_vision_provider_when_enabled(monkeypatch) -> None:
    class FakeVisionClient:
        def __init__(self, api_key: str, model: str):
            assert api_key == "test-key"
            assert model == "gpt-4o-mini"

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            assert analysis_type == "checkup"
            assert image_bytes == b"image"
            assert media_type == "image/png"
            return {
                "analysis_status": "success",
                "extracted_data": {
                    "systolic_bp": 132,
                    "diastolic_bp": 84,
                    "fasting_glucose": 108,
                    "ldl": 132,
                    "hdl": 46,
                },
            }

    monkeypatch.setattr(exam_service, "VisionClient", FakeVisionClient)
    monkeypatch.setattr(exam_service.config, "EXAM_OCR_PROVIDER", "gpt_vision")
    monkeypatch.setattr(exam_service.config, "EXAM_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(exam_service.config, "OPENAI_API_KEY", "test-key")

    result = await exam_service._extract_exam_measurements_with_provider(b"image", "image/png")

    assert result["provider"] == "gpt_vision"
    assert result["fallback_used"] is False
    assert ("systolic_bp", "수축기혈압", "132", "mmHg") in result["measurements"]


def test_convert_exam_pdf_to_png_images_with_pymupdf() -> None:
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "health exam")
    pdf_bytes = document.tobytes()
    document.close()

    result = exam_service._convert_exam_pdf_to_png_images(pdf_bytes)

    assert result.page_count == 1
    assert len(result.images) == 1
    assert result.images[0].startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_exam_ocr_converts_pdf_before_gpt_vision(monkeypatch) -> None:
    calls: list[tuple[bytes, str]] = []
    converted: dict[str, bool] = {}

    class FakeVisionClient:
        def __init__(self, api_key: str, model: str):
            assert api_key == "test-key"
            assert model == "gpt-4o-mini"

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            assert analysis_type == "checkup"
            calls.append((image_bytes, media_type))
            if image_bytes == b"page-1-png":
                return {
                    "analysis_status": "success",
                    "extracted_data": {
                        "systolic_bp": 131,
                        "diastolic_bp": 83,
                    },
                }
            return {
                "analysis_status": "success",
                "extracted_data": {
                    "fasting_glucose": 105,
                },
            }

    def fake_convert(pdf_bytes: bytes):
        assert pdf_bytes == b"%PDF-test"
        converted["called"] = True
        return exam_service.ExamPdfImageConversionResult(page_count=2, images=[b"page-1-png", b"page-2-png"])

    monkeypatch.setattr(exam_service, "VisionClient", FakeVisionClient)
    monkeypatch.setattr(exam_service, "_convert_exam_pdf_to_png_images", fake_convert)
    monkeypatch.setattr(exam_service.config, "EXAM_OCR_PROVIDER", "gpt_vision")
    monkeypatch.setattr(exam_service.config, "EXAM_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(exam_service.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(exam_service.config, "PADDLE_OCR_ENABLED", False)

    result = await exam_service._extract_exam_measurements_with_provider(
        b"%PDF-test",
        "application/pdf",
        "checkup.pdf",
    )

    assert converted["called"] is True
    assert calls == [(b"page-1-png", "image/png"), (b"page-2-png", "image/png")]
    assert result["provider"] == "gpt_vision"
    assert ("systolic_bp", "수축기혈압", "131", "mmHg") in result["measurements"]
    assert ("fasting_glucose", "공복혈당", "105", "mg/dL") in result["measurements"]


@pytest.mark.asyncio
async def test_exam_ocr_pdf_gpt_vision_prioritizes_measurement_page(monkeypatch) -> None:
    calls: list[bytes] = []

    class FakeVisionClient:
        def __init__(self, api_key: str, model: str):
            assert api_key == "test-key"

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            assert analysis_type == "checkup"
            assert media_type == "image/png"
            calls.append(image_bytes)
            assert image_bytes == b"page-2-png"
            return {
                "analysis_status": "success",
                "extracted_data": {
                    "total_cholesterol": "137",
                    "hdl": "49",
                    "triglyceride": "105",
                    "ldl": "67",
                },
            }

    monkeypatch.setattr(exam_service, "VisionClient", FakeVisionClient)
    monkeypatch.setattr(
        exam_service,
        "_convert_exam_pdf_to_png_images",
        lambda _: exam_service.ExamPdfImageConversionResult(
            page_count=3,
            images=[b"page-1-png", b"page-2-png", b"page-3-png"],
        ),
    )
    monkeypatch.setattr(
        exam_service,
        "_extract_exam_pdf_page_texts",
        lambda _: [
            "종합소견 결과 안내",
            "검사항목 결과 참고치 공복혈당 총콜레스테롤 HDL LDL 중성지방 AST ALT 혈색소",
            "발급 안내문 결과",
        ],
    )
    monkeypatch.setattr(exam_service.config, "EXAM_OCR_PROVIDER", "gpt_vision")
    monkeypatch.setattr(exam_service.config, "EXAM_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(exam_service.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(exam_service.config, "PADDLE_OCR_ENABLED", False)

    result = await exam_service._extract_exam_measurements_with_provider(
        b"%PDF-test",
        "application/pdf",
        "checkup.pdf",
    )

    assert calls == [b"page-2-png"]
    assert result["provider"] == "gpt_vision"
    assert ("total_cholesterol", "총콜레스테롤", "137", "mg/dL") in result["measurements"]
    assert ("hdl", "HDL 콜레스테롤", "49", "mg/dL") in result["measurements"]
    assert ("triglyceride", "중성지방", "105", "mg/dL") in result["measurements"]
    assert ("ldl", "LDL 콜레스테롤", "67", "mg/dL") in result["measurements"]


@pytest.mark.asyncio
async def test_exam_ocr_pdf_gpt_vision_falls_back_to_all_pages_when_selected_page_has_no_candidates(
    monkeypatch,
) -> None:
    calls: list[bytes] = []

    class FakeVisionClient:
        def __init__(self, api_key: str, model: str):
            pass

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            calls.append(image_bytes)
            if image_bytes == b"page-3-png":
                return {"extracted_data": {"fasting_glucose": "101"}}
            return {"extracted_data": {}}

    monkeypatch.setattr(exam_service, "VisionClient", FakeVisionClient)
    monkeypatch.setattr(
        exam_service,
        "_convert_exam_pdf_to_png_images",
        lambda _: exam_service.ExamPdfImageConversionResult(
            page_count=3,
            images=[b"page-1-png", b"page-2-png", b"page-3-png"],
        ),
    )
    monkeypatch.setattr(
        exam_service,
        "_extract_exam_pdf_page_texts",
        lambda _: [
            "종합소견 결과 안내",
            "검사항목 결과 참고치 공복혈당 총콜레스테롤 HDL LDL 중성지방 AST ALT 혈색소",
            "발급 안내문 결과",
        ],
    )
    monkeypatch.setattr(exam_service.config, "EXAM_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(exam_service.config, "OPENAI_API_KEY", "test-key")

    result = await exam_service._extract_exam_measurements_with_gpt_vision(
        b"%PDF-test",
        "application/pdf",
        "checkup.pdf",
    )

    assert calls == [b"page-2-png", b"page-1-png", b"page-2-png", b"page-3-png"]
    assert result is not None
    assert result["fallback_used"] is True
    assert ("fasting_glucose", "공복혈당", "101", "mg/dL") in result["measurements"]


@pytest.mark.asyncio
async def test_exam_ocr_returns_empty_measurements_when_provider_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(exam_service.config, "EXAM_OCR_PROVIDER", "gpt_vision")
    monkeypatch.setattr(exam_service.config, "EXAM_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(exam_service.config, "OPENAI_API_KEY", None)

    result = await exam_service._extract_exam_measurements_with_provider(b"image", "image/png")

    assert result["provider"] == "none"
    assert result["fallback_used"] is False
    assert result["measurements"] == []
    assert result["message"] == "인식된 측정값 후보가 없습니다. 파일을 다시 확인해주세요."


@pytest.mark.asyncio
async def test_exam_ocr_returns_empty_measurements_when_parser_extracts_no_values(monkeypatch) -> None:
    class FakeVisionClient:
        def __init__(self, api_key: str, model: str):
            pass

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            return {"analysis_status": "success", "extracted_data": {}}

    monkeypatch.setattr(exam_service, "VisionClient", FakeVisionClient)
    monkeypatch.setattr(exam_service.config, "EXAM_OCR_PROVIDER", "gpt_vision")
    monkeypatch.setattr(exam_service.config, "EXAM_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(exam_service.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(exam_service.config, "PADDLE_OCR_ENABLED", False)

    result = await exam_service._extract_exam_measurements_with_provider(b"image", "image/png")

    assert result["provider"] == "none"
    assert result["fallback_used"] is False
    assert result["measurements"] == []


@pytest.mark.asyncio
async def test_confirm_without_request_updates_latest_health_record_from_existing_measurements(monkeypatch) -> None:
    report = SimpleNamespace(id=1, user_id=10, exam_date=None)
    updated_payload: dict[str, object] = {}

    async def fake_get_exam_report(exam_report_id: int):
        assert exam_report_id == 1
        return report

    async def fake_list_exam_measurements(exam_report_id: int):
        assert exam_report_id == 1
        return [
            _measurement("systolic_bp", "130 mmHg"),
            _measurement("ldl", "131 mg/dL"),
            _measurement("hdl", "47 mg/dL"),
        ]

    async def fake_get_latest_health_record_by_user(user_id: int):
        assert user_id == 10
        return SimpleNamespace(id=99)

    async def fake_update_health_record(record_id: int, data: dict):
        assert record_id == 99
        updated_payload.update(data)
        return SimpleNamespace(id=record_id, **data)

    async def fake_confirm_exam_report(exam_report_id: int):
        assert exam_report_id == 1
        return report

    monkeypatch.setattr(exam_service, "get_exam_report", fake_get_exam_report)
    monkeypatch.setattr(exam_service, "list_exam_measurements", fake_list_exam_measurements)
    monkeypatch.setattr(
        exam_service.health_repository, "get_latest_health_record_by_user", fake_get_latest_health_record_by_user
    )
    monkeypatch.setattr(exam_service.health_repository, "update_health_record", fake_update_health_record)
    monkeypatch.setattr(exam_service.exam_repository, "confirm_exam_report", fake_confirm_exam_report)

    confirmed = await exam_service.confirm_exam_report(1)

    assert confirmed is report
    assert updated_payload == {
        "systolic_bp": 130,
        "ldl_cholesterol": 131,
        "hdl_cholesterol": 47,
    }


def _measurement(key: str, value: object) -> SimpleNamespace:
    return SimpleNamespace(measurement_key=key, value=value)
