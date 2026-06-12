from __future__ import annotations

import pytest

from ai_runtime.ml.inference.screening_predictor import (
    ScreeningArtifactError,
    load_screening_artifact,
    predict_screening_risk,
)


@pytest.mark.parametrize("disease_code", ["HTN", "DM", "DL"])
def test_load_screening_artifact(disease_code: str) -> None:
    artifact = load_screening_artifact(disease_code)

    assert artifact.disease_code == disease_code
    assert artifact.artifact_dir.name == "screening_catboost"
    assert artifact.feature_columns
    assert artifact.threshold > 0
    assert artifact.model_count == 5


@pytest.mark.parametrize("disease_code", ["HTN", "DM", "DL"])
def test_predict_screening_risk_returns_prediction(disease_code: str) -> None:
    artifact = load_screening_artifact(disease_code)
    features = _neutral_features_for(disease_code)

    prediction = predict_screening_risk(disease_code, features)

    assert prediction.disease_code == disease_code
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.threshold == artifact.threshold
    assert isinstance(prediction.screening_high, bool)
    assert prediction.model_count == 5
    assert prediction.missing_features == []
    assert prediction.neutralized_features == []


def test_predict_screening_risk_fills_missing_features() -> None:
    prediction = predict_screening_risk("DM", {"나이": 54, "BMI": 26.5})

    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.model_count == 5
    assert "성별" in prediction.missing_features
    assert "성별" in prediction.neutralized_features
    assert "나이" not in prediction.missing_features
    assert "BMI" not in prediction.missing_features


def test_htn_family_features_are_neutralized_when_missing() -> None:
    prediction = predict_screening_risk("HTN", {"나이": 60, "BMI": 27.0})

    assert "고혈압가족력_부" in prediction.missing_features
    assert "고혈압가족력_부" in prediction.neutralized_features
    assert "고혈압가족력_모" in prediction.neutralized_features
    assert "고혈압가족력_형제" in prediction.neutralized_features


def test_htn_family_features_use_explicit_input_when_present() -> None:
    features = _neutral_features_for("HTN")
    features["고혈압가족력_부"] = 1.0

    prediction = predict_screening_risk("HTN", features)

    assert "고혈압가족력_부" not in prediction.missing_features
    assert "고혈압가족력_부" not in prediction.neutralized_features


def test_unknown_disease_code_raises_clear_error() -> None:
    with pytest.raises(ScreeningArtifactError, match="unsupported screening disease_code"):
        load_screening_artifact("CKD")


def _neutral_features_for(disease_code: str) -> dict[str, float]:
    artifact = load_screening_artifact(disease_code)
    return {column: 0.0 for column in artifact.feature_columns}
