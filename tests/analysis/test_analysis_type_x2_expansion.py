from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from app.dtos.analysis import AnalysisResultCreateRequest, AnalysisRunRequest
from app.models.analysis import AnalysisMode, AnalysisType, RiskLevel

X2_ONLY_ANALYSIS_TYPES = {
    AnalysisType.ABDOMINAL_OBESITY,
    AnalysisType.FATTY_LIVER,
    AnalysisType.ANEMIA,
    AnalysisType.LIVER_FUNCTION,
    AnalysisType.KIDNEY_FUNCTION,
    AnalysisType.CHRONIC_KIDNEY_DISEASE,
}


def test_analysis_type_includes_basic_and_x2_diseases() -> None:
    assert {item.value for item in AnalysisType} == {
        "HYPERTENSION",
        "DIABETES",
        "DYSLIPIDEMIA",
        "OBESITY",
        "ABDOMINAL_OBESITY",
        "FATTY_LIVER",
        "ANEMIA",
        "LIVER_FUNCTION",
        "KIDNEY_FUNCTION",
        "CHRONIC_KIDNEY_DISEASE",
    }


def test_analysis_run_request_accepts_x2_only_analysis_type() -> None:
    request = AnalysisRunRequest(
        analysis_type="CHRONIC_KIDNEY_DISEASE",
        health_record_id=1,
        mode=AnalysisMode.PRECISION,
    )

    assert request.analysis_type == AnalysisType.CHRONIC_KIDNEY_DISEASE


def test_analysis_result_create_request_accepts_x2_only_analysis_type() -> None:
    request = AnalysisResultCreateRequest(
        health_record_id=1,
        analysis_type=AnalysisType.ANEMIA,
        analysis_mode=AnalysisMode.PRECISION,
        risk_score=Decimal("0.00000"),
        risk_level=RiskLevel.LOW,
        analyzed_at=datetime.now(UTC),
    )

    assert request.analysis_type == AnalysisType.ANEMIA
    assert request.analysis_mode == AnalysisMode.PRECISION


def test_longest_analysis_type_fits_migration_column_length() -> None:
    assert max(len(item.value) for item in AnalysisType) == 22
    assert len(AnalysisType.CHRONIC_KIDNEY_DISEASE.value) == 22


def test_analysis_type_migration_expands_column_and_comment() -> None:
    migration_path = Path("app/core/db/migrations/models/8_20260611090000_expand_analysis_type_x2_diseases.py")
    migration_text = migration_path.read_text()

    assert 'ALTER COLUMN "analysis_type" TYPE VARCHAR(22)' in migration_text
    for analysis_type in X2_ONLY_ANALYSIS_TYPES:
        assert f"{analysis_type.value}: {analysis_type.value}" in migration_text
