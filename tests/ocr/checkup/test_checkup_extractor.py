from __future__ import annotations

import pytest

from ai_runtime.ocr.checkup import extractor, pdf_handler, preprocessor
from ai_runtime.ocr.checkup.extractor import (
    parse_blood_pressure,
    parse_from_text_lines,
    parse_height_weight,
    select_measurement_page_lines,
)
from ai_runtime.ocr.checkup.pdf_handler import PdfType


def test_get_ocr_engine_requires_paddleocr_dependency(monkeypatch) -> None:
    monkeypatch.setattr(extractor, "_ocr_engine", None)
    monkeypatch.setattr(extractor, "PaddleOCR", None)

    with pytest.raises(extractor.CheckupOcrDependencyError, match="paddleocr is required"):
        extractor.get_ocr_engine()


def test_get_ocr_engine_uses_monkeypatched_paddleocr(monkeypatch) -> None:
    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(extractor, "_ocr_engine", None)
    monkeypatch.setattr(extractor, "PaddleOCR", FakePaddleOCR)

    engine = extractor.get_ocr_engine()

    assert isinstance(engine, FakePaddleOCR)
    assert engine.kwargs == {"lang": "korean", "use_textline_orientation": True}


def test_pdf_to_images_requires_pdf2image_dependency(monkeypatch) -> None:
    monkeypatch.setattr(pdf_handler, "pdf2image", None)

    with pytest.raises(pdf_handler.PdfImageDependencyError, match="pdf2image is required"):
        pdf_handler.pdf_to_images(b"%PDF")


def test_pdf_to_images_uses_monkeypatched_pdf2image(monkeypatch) -> None:
    class FakeImage:
        width = 10
        height = 20

        def save(self, buffer, *, format, quality) -> None:
            assert format == "JPEG"
            assert quality == 95
            buffer.write(b"image-bytes")

    class FakePdf2Image:
        @staticmethod
        def convert_from_bytes(pdf_bytes, **options):
            assert pdf_bytes == b"%PDF"
            assert options["dpi"] == 200
            return [FakeImage()]

    monkeypatch.setattr(pdf_handler, "pdf2image", FakePdf2Image)

    assert pdf_handler.pdf_to_images(b"%PDF") == [b"image-bytes"]


def test_preprocess_for_ocr_requires_opencv_dependency(monkeypatch) -> None:
    monkeypatch.setattr(preprocessor, "cv2", None)

    with pytest.raises(preprocessor.OpenCvDependencyError, match="opencv-python is required"):
        preprocessor.preprocess_for_ocr(b"image")


def test_parse_height_weight_does_not_reuse_150cm_as_weight() -> None:
    height, weight = parse_height_weight([("신장 150cm 체중 55kg", 0.99)])

    assert height == 150
    assert weight == 55


def test_parse_height_weight_keeps_high_weight_candidate() -> None:
    height, weight = parse_height_weight([("신장 165cm 체중 120kg", 0.99)])

    assert height == 165
    assert weight == 120


def test_parse_blood_pressure_orders_systolic_and_diastolic() -> None:
    systolic, diastolic = parse_blood_pressure([("혈압 80/120 mmHg", 0.99)])

    assert systolic == 120
    assert diastolic == 80


def test_parse_from_text_lines_marks_low_confidence_hb() -> None:
    data, low_conf, _ = parse_from_text_lines([("혈색소 13.2 g/dL", 0.5)])

    assert data.hb == 13.2
    assert "hb" in low_conf


def test_parse_from_text_lines_does_not_mark_high_confidence_hb() -> None:
    data, low_conf, _ = parse_from_text_lines([("혈색소 13.2 g/dL", 0.95)])

    assert data.hb == 13.2
    assert "hb" not in low_conf


def test_select_measurement_page_lines_prioritizes_checkup_table_page() -> None:
    selected = select_measurement_page_lines(
        [
            [("종합소견 결과 안내 HDL", 1.0)],
            [("검사항목 결과 참고치 공복혈당 총콜레스테롤 HDL LDL 중성지방 AST ALT 혈색소", 1.0)],
            [("발급 안내문 결과", 1.0)],
        ]
    )

    assert selected == [("검사항목 결과 참고치 공복혈당 총콜레스테롤 HDL LDL 중성지방 AST ALT 혈색소", 1.0)]


def test_parse_from_text_lines_numeric_value_wins_over_not_applicable_marker() -> None:
    data, _, _ = parse_from_text_lines(
        [
            ("이상지질혈증 비해당", 1.0),
            ("총콜레스테롤 137", 1.0),
            ("HDL 콜레스테롤 49", 1.0),
            ("중성지방 105", 1.0),
            ("LDL 콜레스테롤 67", 1.0),
        ]
    )

    assert data.total_cholesterol == 137
    assert data.hdl == 49
    assert data.triglyceride == 105
    assert data.ldl == 67


def test_parse_from_text_lines_skips_not_applicable_lipid_values() -> None:
    data, _, _ = parse_from_text_lines(
        [
            ("총콜레스테롤 비해당", 1.0),
            ("HDL 콜레스테롤 해당없음", 1.0),
            ("중성지방 검사안함", 1.0),
            ("LDL 콜레스테롤 미실시", 1.0),
        ]
    )

    assert data.total_cholesterol is None
    assert data.hdl is None
    assert data.triglyceride is None
    assert data.ldl is None


@pytest.mark.asyncio
async def test_run_ocr_on_pdf_extracts_lipid_values_from_second_measurement_page(monkeypatch) -> None:
    monkeypatch.setattr(extractor, "detect_pdf_type", lambda _: PdfType.TEXT)
    monkeypatch.setattr(
        extractor,
        "extract_text_from_pdf",
        lambda _: [
            "종합소견 결과 안내 HDL",
            "\n".join(
                [
                    "검사항목 결과 참고치",
                    "공복혈당 91",
                    "총콜레스테롤 137",
                    "HDL 콜레스테롤 49",
                    "중성지방 105",
                    "LDL 콜레스테롤 67",
                    "AST 20 ALT 18 혈색소 14.2",
                ]
            ),
            "발급 안내문 결과",
        ],
    )

    data, _low_confidence_fields, _raw_texts, status = await extractor.run_ocr_on_pdf(b"%PDF")

    assert status != "failed"
    assert data.total_cholesterol == 137
    assert data.hdl == 49
    assert data.triglyceride == 105
    assert data.ldl == 67
