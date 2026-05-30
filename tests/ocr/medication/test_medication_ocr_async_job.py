from pathlib import Path
from types import SimpleNamespace

import pytest

from app.dtos.medications import MedicationOCRResponse
from app.services import medications as medication_service


@pytest.mark.asyncio
async def test_run_medication_ocr_from_job_loads_uploaded_file(monkeypatch, tmp_path) -> None:
    upload_path = tmp_path / "prescription.jpg"
    upload_path.write_bytes(b"image-bytes")
    captured: dict[str, object] = {}

    async def fake_get_job(job_id: int):
        assert job_id == 77
        return SimpleNamespace(
            request_payload={
                "source_type": "MEDICATION_BAG",
                "upload_path": str(upload_path),
                "image_media_type": "image/jpeg",
                "image_filename": "prescription.jpg",
            }
        )

    async def fake_run_medication_ocr(request, image_bytes=None, image_media_type=None):
        captured["source_type"] = request.source_type
        captured["image_filename"] = request.image_filename
        captured["image_bytes"] = image_bytes
        captured["image_media_type"] = image_media_type
        return MedicationOCRResponse(
            source_type=request.source_type,
            ocr_confidence=0.0,
            items=[],
            message="ok",
        )

    monkeypatch.setattr("app.services.async_jobs.get_job", fake_get_job)
    monkeypatch.setattr(medication_service, "run_medication_ocr", fake_run_medication_ocr)

    response = await medication_service.run_medication_ocr_from_job(77)

    assert response.source_type == "MEDICATION_BAG"
    assert captured == {
        "source_type": "MEDICATION_BAG",
        "image_filename": "prescription.jpg",
        "image_bytes": b"image-bytes",
        "image_media_type": "image/jpeg",
    }


def test_medication_ocr_upload_key_uses_user_and_unique_segment() -> None:
    key = medication_service._build_medication_ocr_key(user_id=9, suffix=".jpg")

    assert key.startswith("medication-ocr/9/")
    assert key.endswith("/source.jpg")
    assert ".." not in key


def test_store_medication_ocr_upload_uses_local_storage_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(medication_service.config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(medication_service.config, "LOCAL_STORAGE_ROOT", str(tmp_path))

    stored = medication_service.store_medication_ocr_upload(
        user_id=9,
        image_bytes=b"medication-image",
        image_media_type="image/png",
        filename="bag.png",
    )

    stored_key = stored["upload_path"]
    assert stored_key.startswith("medication-ocr/9/")
    assert stored_key.endswith("/source.png")
    assert (tmp_path / stored_key).read_bytes() == b"medication-image"
    assert stored["image_media_type"] == "image/png"
    assert stored["image_filename"] == "bag.png"


@pytest.mark.asyncio
async def test_run_medication_ocr_from_job_loads_storage_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stored_key = "medication-ocr/9/test/source.webp"
    storage_path = tmp_path / stored_key
    storage_path.parent.mkdir(parents=True)
    storage_path.write_bytes(b"stored-image")
    captured: dict[str, object] = {}

    monkeypatch.setattr(medication_service.config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(medication_service.config, "LOCAL_STORAGE_ROOT", str(tmp_path))

    async def fake_get_job(job_id: int):
        assert job_id == 79
        return SimpleNamespace(
            request_payload={
                "source_type": "MEDICATION_BAG",
                "upload_path": stored_key,
                "image_filename": "bag.webp",
            }
        )

    async def fake_run_medication_ocr(request, image_bytes=None, image_media_type=None):
        captured["source_type"] = request.source_type
        captured["image_filename"] = request.image_filename
        captured["image_bytes"] = image_bytes
        captured["image_media_type"] = image_media_type
        return MedicationOCRResponse(
            source_type=request.source_type,
            ocr_confidence=0.0,
            items=[],
            message="ok",
        )

    monkeypatch.setattr("app.services.async_jobs.get_job", fake_get_job)
    monkeypatch.setattr(medication_service, "run_medication_ocr", fake_run_medication_ocr)

    response = await medication_service.run_medication_ocr_from_job(79)

    assert response.source_type == "MEDICATION_BAG"
    assert captured == {
        "source_type": "MEDICATION_BAG",
        "image_filename": "bag.webp",
        "image_bytes": b"stored-image",
        "image_media_type": "image/webp",
    }


@pytest.mark.asyncio
async def test_run_medication_ocr_from_job_rejects_missing_source(monkeypatch) -> None:
    async def fake_get_job(job_id: int):
        assert job_id == 78
        return SimpleNamespace(request_payload={"source_type": "PRESCRIPTION"})

    monkeypatch.setattr("app.services.async_jobs.get_job", fake_get_job)

    with pytest.raises(ValueError, match="medication_ocr_source_missing"):
        await medication_service.run_medication_ocr_from_job(78)
