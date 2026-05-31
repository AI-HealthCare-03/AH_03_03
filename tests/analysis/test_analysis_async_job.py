from types import SimpleNamespace

import pytest

from app.apis.v1 import analysis_routers
from app.dtos.analysis import AnalysisRunByHealthRecordRequest
from app.models.analysis import AnalysisMode


@pytest.mark.asyncio
async def test_run_analysis_async_endpoint_creates_job(monkeypatch) -> None:
    health_record = SimpleNamespace(id=88, user_id=12)
    user = SimpleNamespace(id=12)
    captured: dict[str, object] = {}

    async def fake_get_health_record(health_record_id: int):
        assert health_record_id == 88
        return health_record

    async def fake_get_missing_fields_for_mode(user_arg, health_record_arg, mode):
        assert user_arg is user
        assert health_record_arg is health_record
        assert mode == AnalysisMode.PRECISION
        return []

    async def fake_create_analysis_run_job(user_id: int, health_record_id: int, mode: str):
        captured.update({"user_id": user_id, "health_record_id": health_record_id, "mode": mode})
        return SimpleNamespace(
            id=33,
            job_type="analysis.run",
            status="PENDING",
            request_payload={},
            result_payload=None,
            error_message=None,
            stream_id="33-0",
            created_at=None,
            updated_at=None,
            started_at=None,
            finished_at=None,
        )

    monkeypatch.setattr(analysis_routers.health_service, "get_health_record", fake_get_health_record)
    monkeypatch.setattr(
        analysis_routers.analysis_service, "get_missing_fields_for_mode", fake_get_missing_fields_for_mode
    )
    monkeypatch.setattr(analysis_routers.async_job_service, "create_analysis_run_job", fake_create_analysis_run_job)

    response = await analysis_routers.run_analysis_async(
        AnalysisRunByHealthRecordRequest(health_record_id=88, mode=AnalysisMode.PRECISION),
        user,
    )

    assert response.job_type == "analysis.run"
    assert captured == {"user_id": 12, "health_record_id": 88, "mode": "PRECISION"}
