from __future__ import annotations

import importlib

import pytest
from fastapi import HTTPException

from ai_runtime.common import image_normalizer
from ai_runtime.common.image_normalizer import (
    HEIC_CONVERSION_ERROR_MESSAGE,
    ImageNormalizationError,
    normalize_upload_image,
)
from ai_runtime.cv import router as cv_router
from app.apis.v1 import diet_routers, exam_routers, medication_routers, upload_routers

checkup_router_module = importlib.import_module("ai_runtime.ocr.checkup.router")


class FakeUpload:
    def __init__(self, filename: str, content_type: str, data: bytes = b"heic-bytes") -> None:
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


class FakeMultipartRequest:
    headers = {"content-type": "multipart/form-data; boundary=test"}

    def __init__(self, form: dict[str, object]) -> None:
        self._form = form

    async def form(self) -> dict[str, object]:
        return self._form


def test_normalize_heic_upload_converts_to_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", lambda image_bytes: b"jpeg-bytes")

    result = normalize_upload_image(b"heic-bytes", "image/heic", "meal.heic")

    assert result.data == b"jpeg-bytes"
    assert result.media_type == "image/jpeg"
    assert result.converted is True
    assert result.original_media_type == "image/heic"


def test_normalize_heif_extension_converts_even_when_mime_is_octet_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", lambda image_bytes: b"jpeg-bytes")

    result = normalize_upload_image(b"heif-bytes", "application/octet-stream", "checkup.heif")

    assert result.data == b"jpeg-bytes"
    assert result.media_type == "image/jpeg"
    assert result.converted is True


def test_normalize_jpeg_and_pdf_keep_existing_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(image_bytes: bytes) -> bytes:
        raise AssertionError("HEIC converter should not run for JPG/PDF")

    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", fail_if_called)

    jpg = normalize_upload_image(b"jpg-bytes", "image/jpeg", "meal.jpg")
    pdf = normalize_upload_image(b"pdf-bytes", "application/pdf", "report.pdf")

    assert jpg.data == b"jpg-bytes"
    assert jpg.media_type == "image/jpeg"
    assert jpg.converted is False
    assert pdf.data == b"pdf-bytes"
    assert pdf.media_type == "application/pdf"
    assert pdf.converted is False


def test_normalize_heic_failure_raises_user_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_conversion(image_bytes: bytes) -> bytes:
        raise ImageNormalizationError(HEIC_CONVERSION_ERROR_MESSAGE)

    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", fail_conversion)

    with pytest.raises(ImageNormalizationError, match="HEIC 이미지를 처리하지 못했습니다"):
        normalize_upload_image(b"broken-heic", "image/heic", "broken.heic")


def test_preview_normalize_heic_returns_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", lambda image_bytes: b"jpeg-preview")

    result = upload_routers._normalize_preview_image(b"heic-bytes", "image/heic", "meal.heic")

    assert result.data == b"jpeg-preview"
    assert result.media_type == "image/jpeg"


def test_preview_normalize_jpeg_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(image_bytes: bytes) -> bytes:
        raise AssertionError("HEIC converter should not run for JPG previews")

    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", fail_if_called)

    result = upload_routers._normalize_preview_image(b"jpg-bytes", "image/jpeg", "meal.jpg")

    assert result.data == b"jpg-bytes"
    assert result.media_type == "image/jpeg"


def test_preview_normalize_rejects_pdf() -> None:
    with pytest.raises(HTTPException) as exc_info:
        upload_routers._normalize_preview_image(b"pdf-bytes", "application/pdf", "report.pdf")

    assert exc_info.value.status_code == 415


@pytest.mark.asyncio
async def test_diet_multipart_heic_is_normalized_before_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", lambda image_bytes: b"jpeg-bytes")
    request = FakeMultipartRequest(
        {
            "description": "점심 식단",
            "image": FakeUpload("meal.heic", "image/heic"),
        }
    )

    payload, image_bytes, image_media_type = await diet_routers._parse_diet_analyze_request(request)

    assert payload.description == "점심 식단"
    assert image_bytes == b"jpeg-bytes"
    assert image_media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_exam_multipart_heic_is_normalized_but_pdf_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", lambda image_bytes: b"jpeg-bytes")
    heic_request = FakeMultipartRequest({"image": FakeUpload("checkup.heic", "image/heic")})
    pdf_request = FakeMultipartRequest({"file": FakeUpload("checkup.pdf", "application/pdf", b"pdf-bytes")})

    heic_bytes, heic_media_type, heic_filename = await exam_routers._read_optional_upload(heic_request)
    pdf_bytes, pdf_media_type, pdf_filename = await exam_routers._read_optional_upload(pdf_request)

    assert heic_bytes == b"jpeg-bytes"
    assert heic_media_type == "image/jpeg"
    assert heic_filename == "checkup.heic"
    assert pdf_bytes == b"pdf-bytes"
    assert pdf_media_type == "application/pdf"
    assert pdf_filename == "checkup.pdf"


@pytest.mark.asyncio
async def test_medication_multipart_heic_is_normalized_before_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", lambda image_bytes: b"jpeg-bytes")
    request = FakeMultipartRequest(
        {
            "source_type": "MEDICATION_BAG",
            "image": FakeUpload("medication.heif", "application/octet-stream"),
        }
    )

    payload, image_bytes, image_media_type = await medication_routers._parse_medication_ocr_request(request)

    assert payload.source_type == "MEDICATION_BAG"
    assert image_bytes == b"jpeg-bytes"
    assert image_media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_runtime_cv_router_normalizes_heic_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", lambda image_bytes: b"jpeg-bytes")
    upload = FakeUpload("meal.heic", "application/octet-stream", b"x" * 1024)

    normalized = await cv_router.validate_image(upload)

    assert normalized.data == b"jpeg-bytes"
    assert normalized.media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_runtime_checkup_router_normalizes_heif_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", lambda image_bytes: b"jpeg-bytes")
    upload = FakeUpload("checkup.heif", "application/octet-stream", b"x" * 1024)

    normalized = await checkup_router_module.validate_image(upload)

    assert normalized.data == b"jpeg-bytes"
    assert normalized.media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_diet_router_returns_415_when_heic_conversion_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_conversion(image_bytes: bytes) -> bytes:
        raise ImageNormalizationError(HEIC_CONVERSION_ERROR_MESSAGE)

    monkeypatch.setattr(image_normalizer, "_convert_heic_to_jpeg", fail_conversion)
    request = FakeMultipartRequest({"image": FakeUpload("broken.heic", "image/heic")})

    with pytest.raises(HTTPException) as exc_info:
        await diet_routers._parse_diet_analyze_request(request)

    assert exc_info.value.status_code == 415
    assert exc_info.value.detail == HEIC_CONVERSION_ERROR_MESSAGE
