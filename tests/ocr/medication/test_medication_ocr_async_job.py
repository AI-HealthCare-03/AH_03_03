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


@pytest.mark.asyncio
async def test_run_medication_ocr_from_job_rejects_missing_source(monkeypatch) -> None:
    async def fake_get_job(job_id: int):
        assert job_id == 78
        return SimpleNamespace(request_payload={"source_type": "PRESCRIPTION"})

    monkeypatch.setattr("app.services.async_jobs.get_job", fake_get_job)

    with pytest.raises(ValueError, match="medication_ocr_source_missing"):
        await medication_service.run_medication_ocr_from_job(78)
