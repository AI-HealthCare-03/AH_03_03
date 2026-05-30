from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_runtime.ml.inference import disease_risk_service
from ai_runtime.ml.inference.catboost_predictor import (
    CatBoostDiseasePredictor,
    FeatureSchemaError,
    _validate_feature_schema,
)
from ai_runtime.ml.inference.disease_risk_service import warmup_chronic_disease_models
from ai_runtime.ml.inference.schemas import DiseasePrediction


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


def test_predict_chronic_disease_risks_reuses_cached_predictor_per_disease(monkeypatch) -> None:
    created: list[str] = []
    warmups: list[str] = []

    class FakePredictor:
        available = True

        def __init__(self, disease: str, artifact_dir):
            _ = artifact_dir
            self.disease = disease
            created.append(disease)

        def load_feature_columns(self) -> list[str]:
            return ["feature_a"]

        def warmup(self) -> int:
            warmups.append(self.disease)
            return 1

        def predict(self, features: dict[str, float]) -> DiseasePrediction:
            assert features == {"feature_a": 1.0}
            return DiseasePrediction(
                disease=self.disease,
                probability=0.2,
                threshold=0.5,
                risk_level="LOW",
                model_name="catboost",
                model_version=f"{self.disease.lower()}-test",
                artifact_dir="test-artifact",
            )

    def fake_map_service_features(user, health_record, feature_columns, strict):
        _ = user, health_record, strict
        assert feature_columns == ["feature_a"]
        return SimpleNamespace(features={"feature_a": 1.0})

    disease_risk_service._PREDICTOR_CACHE.clear()
    disease_risk_service._WARMED_DISEASE_KEYS.clear()
    monkeypatch.setattr(disease_risk_service, "CatBoostDiseasePredictor", FakePredictor)
    monkeypatch.setattr(disease_risk_service, "map_service_features", fake_map_service_features)

    try:
        first = disease_risk_service.predict_chronic_disease_risks(object(), object(), diseases=["DM"])
        second = disease_risk_service.predict_chronic_disease_risks(object(), object(), diseases=["DM"])

        assert created == ["DM"]
        assert warmups == ["DM"]
        assert first["DM"].model_version == "dm-test"
        assert second["DM"].model_version == "dm-test"
    finally:
        disease_risk_service._PREDICTOR_CACHE.clear()
        disease_risk_service._WARMED_DISEASE_KEYS.clear()
