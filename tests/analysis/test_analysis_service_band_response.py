from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from app.apis.v1 import analysis_routers
from app.dtos.analysis import AnalysisResultResponse
from app.models.analysis import AnalysisMode, AnalysisType, RiskLevel
from app.services import analysis as analysis_service


def _analysis_result(**overrides: Any) -> SimpleNamespace:
    values = {
        "id": 11,
        "user_id": 7,
        "health_record_id": 101,
        "async_job_id": None,
        "analysis_type": AnalysisType.HYPERTENSION,
        "analysis_mode": AnalysisMode.BASIC,
        "risk_score": Decimal("0.52000"),
        "risk_level": RiskLevel.MEDIUM,
        "summary": "간편 분석 참고용입니다.",
        "model_name": "rule_based",
        "model_version": "web-basic-v1",
        "analyzed_at": datetime(2026, 6, 11, tzinfo=UTC),
        "created_at": datetime(2026, 6, 11, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 11, tzinfo=UTC),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_analysis_result_response_extracts_service_band_from_snapshot() -> None:
    result = _analysis_result()
    snapshot = SimpleNamespace(
        output_payload={
            "final_outputs": {
                "risk_level": "MEDIUM",
                "service_band": "CAUTION",
                "service_band_label": "주의",
                "service_band_percent": 65,
                "legacy_risk_level": "MEDIUM",
                "screening_probability": 0.91,
            }
        }
    )

    payload = analysis_service._analysis_result_response(result, snapshot)

    assert payload["risk_level"] == RiskLevel.MEDIUM
    assert payload["service_band"] == "CAUTION"
    assert payload["service_band_label"] == "주의"
    assert payload["service_band_percent"] == 65
    assert payload["legacy_risk_level"] == "MEDIUM"
    assert "screening_probability" not in payload
    AnalysisResultResponse.model_validate(payload)


def test_analysis_result_response_allows_legacy_result_without_service_band() -> None:
    payload = analysis_service._analysis_result_response(_analysis_result(), None)

    assert payload["risk_level"] == RiskLevel.MEDIUM
    assert payload["service_band"] is None
    assert payload["service_band_label"] is None
    assert payload["service_band_percent"] is None
    assert payload["legacy_risk_level"] is None
    AnalysisResultResponse.model_validate(payload)


@pytest.mark.asyncio
async def test_list_analysis_results_router_returns_optional_service_band(monkeypatch: pytest.MonkeyPatch) -> None:
    user = SimpleNamespace(id=7)
    result = _analysis_result()
    enriched_payload = analysis_service._analysis_result_response(
        result,
        SimpleNamespace(
            output_payload={
                "final_outputs": {
                    "service_band": "ATTENTION",
                    "service_band_label": "관심 필요",
                    "service_band_percent": 45,
                    "legacy_risk_level": "MEDIUM",
                }
            }
        ),
    )

    async def fake_safe_record_sensitive_access(**kwargs: Any) -> None:
        _ = kwargs

    async def fake_list_analysis_results(user_id: int, limit: int = 20, offset: int = 0) -> list[Any]:
        assert user_id == 7
        assert limit == 20
        assert offset == 0
        return [result]

    async def fake_list_analysis_result_responses(results: list[Any]) -> list[dict[str, Any]]:
        assert results == [result]
        return [enriched_payload]

    monkeypatch.setattr(analysis_routers, "safe_record_sensitive_access", fake_safe_record_sensitive_access)
    monkeypatch.setattr(analysis_routers.analysis_service, "list_analysis_results", fake_list_analysis_results)
    monkeypatch.setattr(
        analysis_routers.analysis_service,
        "list_analysis_result_responses",
        fake_list_analysis_result_responses,
    )

    response = await analysis_routers.list_analysis_results(SimpleNamespace(), user)

    assert response[0]["service_band"] == "ATTENTION"
    assert response[0]["service_band_label"] == "관심 필요"
    assert response[0]["service_band_percent"] == 45
    assert "probability" not in response[0]


@pytest.mark.asyncio
async def test_get_analysis_result_detail_uses_enriched_result_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _analysis_result()

    async def fake_get_analysis_result(result_id: int) -> Any:
        assert result_id == 11
        return result

    async def fake_list_analysis_factors(result_id: int) -> list[Any]:
        assert result_id == 11
        return []

    async def fake_get_analysis_snapshot(result_id: int) -> Any:
        assert result_id == 11
        return SimpleNamespace(
            output_payload={
                "final_outputs": {
                    "service_band": "HIGH_CAUTION",
                    "service_band_label": "높은 주의",
                    "service_band_percent": 80,
                    "legacy_risk_level": "HIGH",
                }
            }
        )

    monkeypatch.setattr(analysis_service, "get_analysis_result", fake_get_analysis_result)
    monkeypatch.setattr(analysis_service, "list_analysis_factors", fake_list_analysis_factors)
    monkeypatch.setattr(analysis_service, "get_analysis_snapshot", fake_get_analysis_snapshot)
    monkeypatch.setattr(analysis_service, "_analysis_explanation", lambda result, factors: {})

    detail = await analysis_service.get_analysis_result_detail(11)

    assert detail is not None
    assert detail["result"]["service_band"] == "HIGH_CAUTION"
    assert detail["result"]["service_band_label"] == "높은 주의"
    assert detail["result"]["service_band_percent"] == 80
    assert "probability" not in detail["result"]
