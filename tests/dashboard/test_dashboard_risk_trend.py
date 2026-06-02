from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

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
            risk_level=RiskLevel.MEDIUM,
        ),
        SimpleNamespace(
            analysis_type=AnalysisType.HYPERTENSION,
            analyzed_at=now,
            risk_score=Decimal("0.56000"),
            risk_level=RiskLevel.HIGH,
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
    assert result["series"][1]["points"][0]["risk_level"] == RiskLevel.HIGH
