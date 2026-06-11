from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from ai_runtime.ml.inference import screening_risk_service
from ai_runtime.ml.inference.dual_stage_policy import ServiceBand
from ai_runtime.ml.inference.screening_predictor import ScreeningPrediction
from ai_runtime.ml.inference.screening_risk_service import (
    ScreeningRiskServiceError,
    UnsupportedScreeningRiskDiseaseError,
    predict_screening_dual_stage_risk,
)


@pytest.mark.parametrize(
    ("base_high", "screening_high", "expected_band", "expected_label", "expected_percent", "expected_legacy"),
    [
        (False, False, ServiceBand.LOW, "낮음", 25, "LOW"),
        (True, False, ServiceBand.ATTENTION, "관심 필요", 45, "MEDIUM"),
        (False, True, ServiceBand.CAUTION, "주의", 65, "MEDIUM"),
        (True, True, ServiceBand.HIGH_CAUTION, "높은 주의", 80, "HIGH"),
    ],
)
def test_predict_screening_dual_stage_risk_policy_combinations(
    monkeypatch: pytest.MonkeyPatch,
    base_high: bool,
    screening_high: bool,
    expected_band: ServiceBand,
    expected_label: str,
    expected_percent: int,
    expected_legacy: str,
) -> None:
    _patch_screening_prediction(monkeypatch, screening_high=screening_high)

    result = predict_screening_dual_stage_risk(
        disease_code="HTN",
        features={"나이": 55},
        base_high=base_high,
    )

    assert result.disease_code == "HTN"
    assert result.base_high is base_high
    assert result.screening_high is screening_high
    assert result.service_band == expected_band
    assert result.service_band_label == expected_label
    assert result.service_band_percent == expected_percent
    assert result.legacy_risk_level == expected_legacy
    assert result.screening_model_count == 5


@pytest.mark.parametrize(
    ("base_risk_level", "expected_base_high", "expected_legacy"),
    [
        ("LOW", False, "LOW"),
        ("MEDIUM", True, "MEDIUM"),
        ("HIGH", True, "MEDIUM"),
    ],
)
def test_base_risk_level_to_base_high_conversion(
    monkeypatch: pytest.MonkeyPatch,
    base_risk_level: str,
    expected_base_high: bool,
    expected_legacy: str,
) -> None:
    _patch_screening_prediction(monkeypatch, screening_high=False)

    result = predict_screening_dual_stage_risk(
        disease_code="DM",
        features={},
        base_risk_level=base_risk_level,
    )

    assert result.base_high is expected_base_high
    assert result.legacy_risk_level == expected_legacy


@pytest.mark.parametrize("disease_code", ["HTN", "DM", "DL"])
def test_supported_disease_codes(monkeypatch: pytest.MonkeyPatch, disease_code: str) -> None:
    calls: list[str] = []

    def fake_predict_screening_risk(code: str, features: Mapping[str, Any]) -> ScreeningPrediction:
        calls.append(code)
        assert features == {"BMI": 25.0}
        return _prediction(code, screening_high=False)

    monkeypatch.setattr(
        screening_risk_service.screening_predictor,
        "predict_screening_risk",
        fake_predict_screening_risk,
    )

    result = predict_screening_dual_stage_risk(
        disease_code=disease_code.lower(),
        features={"BMI": 25.0},
        base_risk_level="LOW",
    )

    assert result.disease_code == disease_code
    assert calls == [disease_code]


def test_unknown_disease_code_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_screening_prediction(monkeypatch, screening_high=False)

    with pytest.raises(UnsupportedScreeningRiskDiseaseError, match="unsupported screening risk disease_code"):
        predict_screening_dual_stage_risk(disease_code="CKD", features={}, base_risk_level="LOW")


def test_invalid_base_risk_level_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_screening_prediction(monkeypatch, screening_high=False)

    with pytest.raises(ScreeningRiskServiceError, match="unsupported base_risk_level"):
        predict_screening_dual_stage_risk(disease_code="HTN", features={}, base_risk_level="UNKNOWN")


def test_base_high_or_base_risk_level_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_screening_prediction(monkeypatch, screening_high=False)

    with pytest.raises(ScreeningRiskServiceError, match="base_high or base_risk_level is required"):
        predict_screening_dual_stage_risk(disease_code="HTN", features={})


def test_screening_metadata_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_screening_prediction(
        monkeypatch,
        screening_high=True,
        missing_features=["성별"],
        neutralized_features=["성별"],
    )

    result = predict_screening_dual_stage_risk(
        disease_code="DL",
        features={},
        base_risk_level="LOW",
    )

    assert result.screening_missing_features == ["성별"]
    assert result.screening_neutralized_features == ["성별"]
    assert result.screening_model_count == 5


def test_raw_probability_is_not_user_facing_result_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_screening_prediction(monkeypatch, screening_high=True)

    result = predict_screening_dual_stage_risk(
        disease_code="HTN",
        features={},
        base_risk_level="LOW",
    )

    assert not hasattr(result, "probability")
    assert not hasattr(result, "screening_probability")
    assert "Raw screening probability is intentionally not copied" in (
        screening_risk_service.predict_screening_dual_stage_risk.__doc__ or ""
    )


def _patch_screening_prediction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    screening_high: bool,
    missing_features: list[str] | None = None,
    neutralized_features: list[str] | None = None,
) -> None:
    def fake_predict_screening_risk(disease_code: str, features: Mapping[str, Any]) -> ScreeningPrediction:
        _ = features
        return _prediction(
            disease_code,
            screening_high=screening_high,
            missing_features=missing_features or [],
            neutralized_features=neutralized_features or [],
        )

    monkeypatch.setattr(
        screening_risk_service.screening_predictor,
        "predict_screening_risk",
        fake_predict_screening_risk,
    )


def _prediction(
    disease_code: str,
    *,
    screening_high: bool,
    missing_features: list[str] | None = None,
    neutralized_features: list[str] | None = None,
) -> ScreeningPrediction:
    return ScreeningPrediction(
        disease_code=disease_code,
        probability=0.9 if screening_high else 0.1,
        threshold=0.5,
        screening_high=screening_high,
        missing_features=missing_features or [],
        neutralized_features=neutralized_features or [],
        model_count=5,
    )
