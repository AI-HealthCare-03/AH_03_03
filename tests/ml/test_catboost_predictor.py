from __future__ import annotations

import pytest

from ai_runtime.ml.inference.catboost_predictor import (
    CatBoostDiseasePredictor,
    FeatureSchemaError,
    _validate_feature_schema,
)
from ai_runtime.ml.inference.disease_risk_service import warmup_chronic_disease_models


def test_validate_feature_schema_rejects_missing_feature() -> None:
    with pytest.raises(FeatureSchemaError):
        _validate_feature_schema({"성별": 1.0}, ["성별", "나이"])


def test_validate_feature_schema_rejects_unexpected_feature() -> None:
    with pytest.raises(FeatureSchemaError):
        _validate_feature_schema({"성별": 1.0, "나이": 45.0, "예상외_feature": 0.0}, ["성별", "나이"])


def test_validate_feature_schema_accepts_exact_feature_columns() -> None:
    _validate_feature_schema({"성별": 1.0, "나이": 45.0}, ["성별", "나이"])


def test_predictor_warmup_returns_zero_when_artifact_unavailable(tmp_path) -> None:
    predictor = CatBoostDiseasePredictor("DM", tmp_path)

    assert predictor.warmup() == 0


def test_warmup_chronic_disease_models_reports_unknown_disease() -> None:
    result = warmup_chronic_disease_models(["UNKNOWN"])

    assert result["UNKNOWN"]["status"] == "skipped"
    assert result["UNKNOWN"]["reason"] == "unknown_disease"
