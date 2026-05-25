from __future__ import annotations

from decimal import Decimal
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


@pytest.mark.asyncio
async def test_exam_ocr_marks_fallback_when_provider_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(exam_service.config, "EXAM_OCR_PROVIDER", "gpt_vision")
    monkeypatch.setattr(exam_service.config, "EXAM_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(exam_service.config, "OPENAI_API_KEY", None)

    result = await exam_service._extract_exam_measurements_with_provider(b"image", "image/png")

    assert result["provider"] == "fallback"
    assert result["fallback_used"] is True
    assert result["measurements"] == exam_service.FALLBACK_OCR_MEASUREMENTS


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
