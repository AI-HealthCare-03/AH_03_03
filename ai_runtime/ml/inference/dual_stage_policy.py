from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ServiceBand(StrEnum):
    LOW = "LOW"
    ATTENTION = "ATTENTION"
    CAUTION = "CAUTION"
    HIGH_CAUTION = "HIGH_CAUTION"


SERVICE_BAND_LABELS: dict[ServiceBand, str] = {
    ServiceBand.LOW: "낮음",
    ServiceBand.ATTENTION: "관심 필요",
    ServiceBand.CAUTION: "주의",
    ServiceBand.HIGH_CAUTION: "높은 주의",
}

SERVICE_BAND_PERCENTS: dict[ServiceBand, int] = {
    ServiceBand.LOW: 25,
    ServiceBand.ATTENTION: 45,
    ServiceBand.CAUTION: 65,
    ServiceBand.HIGH_CAUTION: 80,
}


@dataclass(frozen=True)
class DualStagePolicyResult:
    risk_level: str
    service_band: ServiceBand
    service_band_label: str
    service_band_percent: int
    legacy_risk_level: str | None = None


def resolve_dual_stage_band(base_risk_level: ServiceBand | str, screening_high: bool) -> ServiceBand:
    """Resolve X1 base result and screening signal into a promotion-limited band.

    The screening model is an auxiliary signal, not a gate. It may promote
    LOW/ATTENTION up to CAUTION, but it must not promote CAUTION to
    HIGH_CAUTION by itself. HIGH_CAUTION is reserved for base score/rule results
    that are already high.
    """
    base_band = coerce_service_band(base_risk_level)
    if base_band == ServiceBand.HIGH_CAUTION:
        return ServiceBand.HIGH_CAUTION
    if base_band == ServiceBand.CAUTION:
        return ServiceBand.CAUTION
    if screening_high:
        return ServiceBand.CAUTION
    return base_band


def resolve_dual_stage_result(base_risk_level: ServiceBand | str, screening_high: bool) -> DualStagePolicyResult:
    service_band = resolve_dual_stage_band(base_risk_level=base_risk_level, screening_high=screening_high)
    return DualStagePolicyResult(
        risk_level=service_band.value,
        service_band=service_band,
        service_band_label=get_service_band_label(service_band),
        service_band_percent=get_service_band_percent(service_band),
        legacy_risk_level=to_legacy_risk_level(service_band),
    )


def apply_strict_screening_policy_v2(
    *,
    base_risk_level: ServiceBand | str,
    screening_high: bool,
    strict_high: bool,
) -> DualStagePolicyResult:
    """Resolve BASIC rule, screening, and strict model signals into v2 band.

    The strict model is an auxiliary signal only. It can promote CAUTION to
    HIGH_CAUTION, but must not promote LOW/ATTENTION directly to HIGH_CAUTION.
    Screening remains capped at CAUTION.
    """
    base_band = coerce_service_band(base_risk_level)
    if base_band == ServiceBand.HIGH_CAUTION:
        service_band = ServiceBand.HIGH_CAUTION
    elif base_band == ServiceBand.CAUTION and strict_high:
        service_band = ServiceBand.HIGH_CAUTION
    else:
        service_band = resolve_dual_stage_band(
            base_risk_level=base_band,
            screening_high=screening_high,
        )
    return DualStagePolicyResult(
        risk_level=service_band.value,
        service_band=service_band,
        service_band_label=get_service_band_label(service_band),
        service_band_percent=get_service_band_percent(service_band),
        legacy_risk_level=to_legacy_risk_level(service_band),
    )


def coerce_service_band(base_risk_level: ServiceBand | str) -> ServiceBand:
    normalized = str(base_risk_level).upper()
    if normalized == "MEDIUM":
        return ServiceBand.CAUTION
    if normalized == "HIGH":
        return ServiceBand.HIGH_CAUTION
    return ServiceBand(normalized)


def get_service_band_label(service_band: ServiceBand) -> str:
    return SERVICE_BAND_LABELS[service_band]


def get_service_band_percent(service_band: ServiceBand) -> int:
    return SERVICE_BAND_PERCENTS[service_band]


def to_legacy_risk_level(service_band: ServiceBand) -> str:
    """Return the canonical 4-step risk level value.

    The function name is kept temporarily for older call sites/tests. The
    official runtime risk level is now LOW/ATTENTION/CAUTION/HIGH_CAUTION.
    """
    return service_band.value
