from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.analysis import AnalysisType, RiskLevel
from app.services import dashboard as dashboard_service


class FakeAnalysisResultQuery:
    def __init__(self, rows):
        self.rows = rows
        self.filters = {}
        self.ordering = ()
        self.query_limit = None

    def order_by(self, *ordering):
        self.ordering = ordering
        return self

    def limit(self, limit):
        self.query_limit = limit
        return self

    def __await__(self):
        async def _result():
            return self.rows

        return _result().__await__()


@pytest.mark.asyncio
async def test_dashboard_risk_trend_groups_analysis_results_by_disease(monkeypatch) -> None:
    now = datetime.now(UTC)
    rows = [
        SimpleNamespace(
            analysis_type=AnalysisType.DIABETES,
            analyzed_at=now - timedelta(days=2),
            risk_score=Decimal("0.12000"),
            risk_level=RiskLevel.LOW,
        ),
        SimpleNamespace(
            analysis_type=AnalysisType.DIABETES,
            analyzed_at=now - timedelta(days=1),
            risk_score=Decimal("0.34000"),
            risk_level=RiskLevel.CAUTION,
        ),
        SimpleNamespace(
            analysis_type=AnalysisType.HYPERTENSION,
            analyzed_at=now,
            risk_score=Decimal("0.56000"),
            risk_level=RiskLevel.HIGH_CAUTION,
        ),
    ]
    query = FakeAnalysisResultQuery(rows)

    def fake_filter(**filters):
        query.filters = filters
        return query

    monkeypatch.setattr(dashboard_service.AnalysisResult, "filter", fake_filter)

    result = await dashboard_service.get_dashboard_risk_trend(user_id=42, period="all")

    assert query.filters == {"user_id": 42}
    assert query.ordering == ("analysis_type", "analyzed_at")
    assert query.query_limit == 1000
    assert result["period"] == "all"
    assert result["series"][0]["disease_type"] == AnalysisType.DIABETES
    assert [point["risk_score"] for point in result["series"][0]["points"]] == [0.12, 0.34]
    assert result["series"][1]["disease_type"] == AnalysisType.HYPERTENSION
    assert result["series"][1]["points"][0]["risk_level"] == RiskLevel.HIGH_CAUTION
    assert result["series"][1]["points"][0]["service_band"] == "HIGH_CAUTION"


@pytest.mark.asyncio
async def test_dashboard_health_trends_are_oldest_first_with_same_day_records(monkeypatch) -> None:
    day = datetime(2026, 6, 16, tzinfo=UTC)
    newer_record = SimpleNamespace(
        id=3,
        measured_at=day.replace(hour=9),
        created_at=day.replace(hour=9, minute=3),
        fasting_glucose=180,
        systolic_bp=150,
        diastolic_bp=95,
        weight_kg=Decimal("70.0"),
    )
    older_record = SimpleNamespace(
        id=1,
        measured_at=day.replace(hour=8),
        created_at=day.replace(hour=8, minute=1),
        fasting_glucose=95,
        systolic_bp=120,
        diastolic_bp=80,
        weight_kg=Decimal("80.0"),
    )
    tie_break_record = SimpleNamespace(
        id=2,
        measured_at=day.replace(hour=8),
        created_at=day.replace(hour=8, minute=2),
        fasting_glucose=110,
        systolic_bp=130,
        diastolic_bp=85,
        weight_kg=Decimal("78.0"),
    )

    async def fake_list_health_records(user_id: int, limit: int) -> list[Any]:
        assert user_id == 42
        assert limit == 1000
        return [newer_record, tie_break_record, older_record]

    async def fake_list_diet_records(user_id: int, limit: int) -> list[Any]:
        assert user_id == 42
        return []

    async def fake_build_challenge_completion_rates(user_id: int, date_from, date_to) -> list[Any]:
        assert user_id == 42
        return []

    monkeypatch.setattr(dashboard_service.health_service, "list_health_records", fake_list_health_records)
    monkeypatch.setattr(dashboard_service.diet_service, "list_diet_records", fake_list_diet_records)
    monkeypatch.setattr(
        dashboard_service,
        "_build_challenge_completion_rates",
        fake_build_challenge_completion_rates,
    )

    result = await dashboard_service.get_dashboard_trends(user_id=42, period="all")

    assert [point["value"] for point in result["glucose"]] == [95, 110, 180]
    assert [point["systolic"] for point in result["blood_pressure"]] == [120, 130, 150]
    assert [point["value"] for point in result["weight"]] == [80.0, 78.0, 70.0]
    assert [point["id"] for point in result["glucose"]] == [1, 2, 3]
    assert all(point["measured_at"] for point in result["glucose"])
    assert all(point["created_at"] for point in result["glucose"])


def test_dashboard_today_period_uses_current_day(monkeypatch) -> None:
    today = datetime(2026, 6, 18, tzinfo=UTC)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return today.astimezone(tz) if tz is not None else today.replace(tzinfo=None)

    monkeypatch.setattr(dashboard_service, "datetime", FixedDateTime)

    normalized_period, date_from, date_to = dashboard_service.normalize_period("today")

    assert normalized_period == "today"
    assert date_from == today.date()
    assert date_to == today.date()


@pytest.mark.asyncio
async def test_dashboard_medications_returns_json_safe_payload(monkeypatch) -> None:
    now = datetime(2026, 6, 18, 9, 30, tzinfo=UTC)
    medication = SimpleNamespace(
        id=10,
        user_id=42,
        name="비타민 D",
        medication_type="SUPPLEMENT",
        dosage=None,
        frequency="매일 1회",
        reminder_time=time(9, 0),
        is_active=True,
        memo=None,
        created_at=now,
        updated_at=now,
    )
    medication_record = SimpleNamespace(
        id=20,
        medication_id=10,
        user_id=42,
        scheduled_at=None,
        taken_at=now,
        is_taken=True,
        status="TAKEN",
        memo=None,
        created_at=now,
        updated_at=now,
    )

    async def fake_list_medications(user_id: int, is_active: bool, limit: int) -> list[Any]:
        assert user_id == 42
        assert is_active is True
        assert limit == 10
        return [medication]

    async def fake_list_medication_records(user_id: int, limit: int) -> list[Any]:
        assert user_id == 42
        assert limit == 10
        return [medication_record]

    monkeypatch.setattr(dashboard_service.medication_service, "list_medications", fake_list_medications)
    monkeypatch.setattr(
        dashboard_service.medication_service,
        "list_medication_records",
        fake_list_medication_records,
    )

    result = await dashboard_service.get_dashboard_medications(user_id=42)

    medication_payload = result["active_medications"][0]
    assert medication_payload["id"] == 10
    assert medication_payload["name"] == "비타민 D"
    assert medication_payload["dosage"] is None
    assert medication_payload["reminder_time"] == "09:00:00"
    assert isinstance(medication_payload["created_at"], str)

    record_payload = result["recent_medication_records"][0]
    assert record_payload["id"] == 20
    assert record_payload["medication_id"] == 10
    assert record_payload["scheduled_at"] is None
    assert isinstance(record_payload["taken_at"], str)
    assert record_payload["is_taken"] is True


@pytest.mark.asyncio
async def test_dashboard_medications_allows_empty_lists(monkeypatch) -> None:
    async def fake_list_medications(user_id: int, is_active: bool, limit: int) -> list[Any]:
        assert user_id == 42
        assert is_active is True
        assert limit == 10
        return []

    async def fake_list_medication_records(user_id: int, limit: int) -> list[Any]:
        assert user_id == 42
        assert limit == 10
        return []

    monkeypatch.setattr(dashboard_service.medication_service, "list_medications", fake_list_medications)
    monkeypatch.setattr(
        dashboard_service.medication_service,
        "list_medication_records",
        fake_list_medication_records,
    )

    result = await dashboard_service.get_dashboard_medications(user_id=42)

    assert result == {"active_medications": [], "recent_medication_records": []}


@pytest.mark.asyncio
async def test_dashboard_summary_includes_x2_source_detail_fields(monkeypatch) -> None:
    now = datetime.now(UTC)
    analysis_result = SimpleNamespace(
        id=11,
        analysis_type=AnalysisType.HYPERTENSION,
        analysis_mode="PRECISION",
        risk_score=Decimal("0.65000"),
        risk_level=RiskLevel.CAUTION,
    )

    async def fake_get_latest_health_record(user_id: int) -> None:
        assert user_id == 42
        return None

    async def fake_list_unread_notifications(user_id: int, limit: int = 1000) -> list[Any]:
        assert user_id == 42
        assert limit == 1000
        return []

    async def fake_count_active_user_challenges(user_id: int) -> int:
        assert user_id == 42
        return 0

    async def fake_list_medications(user_id: int, is_active: bool, limit: int = 1000) -> list[Any]:
        assert user_id == 42
        assert is_active is True
        assert limit == 1000
        return []

    async def fake_list_latest_analysis_results(user_id: int) -> list[Any]:
        assert user_id == 42
        return [analysis_result]

    async def fake_list_analysis_result_responses(results: list[Any]) -> list[dict[str, Any]]:
        assert results == [analysis_result]
        return [
            {
                "id": 11,
                "analysis_type": AnalysisType.HYPERTENSION,
                "analysis_mode": "PRECISION",
                "risk_level": RiskLevel.CAUTION,
                "risk_score": Decimal("0.65000"),
                "service_band": "CAUTION",
                "service_band_label": "주의",
                "service_band_percent": 65,
                "legacy_risk_level": None,
                "result_source": "X2_RULE",
                "x2_stage_code": "HTN_STAGE_1",
                "x2_stage_label": "고혈압 1단계 범위",
                "x2_available": True,
                "x2_missing_fields": [],
                "selected_exam_report_id": 9001,
                "x2_measurement_source": "exam_measurements",
                "summary": "정밀 분석 참고용입니다.",
                "model_name": "x2_rule",
                "model_version": "x2-rule-v1",
                "analyzed_at": now,
                "created_at": now,
            }
        ]

    async def fake_build_top_risk_factors(results: list[Any]) -> list[Any]:
        assert results == [analysis_result]
        return []

    monkeypatch.setattr(dashboard_service.health_service, "get_latest_health_record", fake_get_latest_health_record)
    monkeypatch.setattr(
        dashboard_service.notification_service,
        "list_unread_notifications",
        fake_list_unread_notifications,
    )
    monkeypatch.setattr(
        dashboard_service.challenge_service,
        "count_active_user_challenges",
        fake_count_active_user_challenges,
    )
    monkeypatch.setattr(dashboard_service.medication_service, "list_medications", fake_list_medications)
    monkeypatch.setattr(
        dashboard_service.analysis_service,
        "list_latest_analysis_results",
        fake_list_latest_analysis_results,
    )
    monkeypatch.setattr(
        dashboard_service.analysis_service,
        "list_analysis_result_responses",
        fake_list_analysis_result_responses,
    )
    monkeypatch.setattr(dashboard_service, "_build_top_risk_factors", fake_build_top_risk_factors)

    result = await dashboard_service.get_dashboard_summary(user_id=42)

    latest = result["latest_analysis_results"][0]
    assert latest["result_source"] == "X2_RULE"
    assert latest["x2_stage_code"] == "HTN_STAGE_1"
    assert latest["x2_stage_label"] == "고혈압 1단계 범위"
    assert latest["x2_available"] is True
    assert latest["x2_missing_fields"] == []
    assert latest["selected_exam_report_id"] == 9001
    assert latest["x2_measurement_source"] == "exam_measurements"
