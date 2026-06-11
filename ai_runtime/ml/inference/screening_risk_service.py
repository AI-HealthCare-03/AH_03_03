from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ai_runtime.ml.inference import screening_predictor
from ai_runtime.ml.inference.dual_stage_policy import (
    ServiceBand,
    resolve_dual_stage_result,
)

SUPPORTED_SCREENING_RISK_DISEASES = frozenset({"HTN", "DM", "DL"})
BASE_HIGH_RISK_LEVELS = frozenset({"MEDIUM", "HIGH"})
BASE_LOW_RISK_LEVELS = frozenset({"LOW"})


class ScreeningRiskServiceError(ValueError):
    pass


class UnsupportedScreeningRiskDiseaseError(ScreeningRiskServiceError):
    pass


@dataclass(frozen=True)
class ScreeningRiskResult:
    disease_code: str
    base_high: bool
    screening_high: bool
    service_band: ServiceBand
    service_band_label: str
    service_band_percent: int
    legacy_risk_level: str
    screening_missing_features: list[str]
    screening_neutralized_features: list[str]
    screening_model_count: int


def predict_screening_dual_stage_risk(
    *,
    disease_code: str,
    features: Mapping[str, Any],
    base_risk_level: str | None = None,
    base_high: bool | None = None,
) -> ScreeningRiskResult:
    """Combine BASIC X1 base risk and screening model signal.

    The returned service band is the intended frontend display value. The
    legacy LOW/MEDIUM/HIGH risk level is kept only for compatibility with the
    current DB/API contract.

    Raw screening probability is intentionally not copied into this service
    result. It may be used for internal validation/debugging, but user-facing
    DTOs should expose the 4-step service band label and percent instead.
    """
    normalized_disease_code = _normalize_disease_code(disease_code)
    resolved_base_high = _resolve_base_high(base_risk_level=base_risk_level, base_high=base_high)
    screening_result = screening_predictor.predict_screening_risk(normalized_disease_code, features)
    policy_result = resolve_dual_stage_result(
        base_high=resolved_base_high,
        screening_high=screening_result.screening_high,
    )

    return ScreeningRiskResult(
        disease_code=normalized_disease_code,
        base_high=resolved_base_high,
        screening_high=screening_result.screening_high,
        service_band=policy_result.service_band,
        service_band_label=policy_result.service_band_label,
        service_band_percent=policy_result.service_band_percent,
        legacy_risk_level=policy_result.legacy_risk_level,
        screening_missing_features=screening_result.missing_features,
        screening_neutralized_features=screening_result.neutralized_features,
        screening_model_count=screening_result.model_count,
    )


def _normalize_disease_code(disease_code: str) -> str:
    normalized = disease_code.upper()
    if normalized not in SUPPORTED_SCREENING_RISK_DISEASES:
        supported = ", ".join(sorted(SUPPORTED_SCREENING_RISK_DISEASES))
        raise UnsupportedScreeningRiskDiseaseError(
            f"unsupported screening risk disease_code={disease_code!r}; supported={supported}"
        )
    return normalized


def _resolve_base_high(*, base_risk_level: str | None, base_high: bool | None) -> bool:
    if base_high is not None:
        return base_high
    if base_risk_level is None:
        raise ScreeningRiskServiceError("base_high or base_risk_level is required")

    normalized_risk_level = base_risk_level.upper()
    if normalized_risk_level in BASE_HIGH_RISK_LEVELS:
        return True
    if normalized_risk_level in BASE_LOW_RISK_LEVELS:
        return False
    raise ScreeningRiskServiceError(f"unsupported base_risk_level={base_risk_level!r}")
