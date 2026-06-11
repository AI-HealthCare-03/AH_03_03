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

# Compatibility mapping for the existing LOW/MEDIUM/HIGH DB/API contract.
# Do not use this value as the primary frontend display band.
LEGACY_RISK_LEVELS: dict[ServiceBand, str] = {
    ServiceBand.LOW: "LOW",
    ServiceBand.ATTENTION: "MEDIUM",
    ServiceBand.CAUTION: "MEDIUM",
    ServiceBand.HIGH_CAUTION: "HIGH",
}


@dataclass(frozen=True)
class DualStagePolicyResult:
    service_band: ServiceBand
    service_band_label: str
    service_band_percent: int
    legacy_risk_level: str


def resolve_dual_stage_band(base_high: bool, screening_high: bool) -> ServiceBand:
    """Resolve X1 base rule and screening signal into a conservative service band.

    The screening model is an auxiliary signal, not a gate. A high base result
    must never be downgraded to LOW just because screening is low. Real-data
    comparison showed the screening-high group had higher label-positive rates,
    so screening-high/base-low is promoted to CAUTION while base-high/screening-low
    remains ATTENTION.
    """
    if base_high and screening_high:
        return ServiceBand.HIGH_CAUTION
    if base_high and not screening_high:
        return ServiceBand.ATTENTION
    if not base_high and screening_high:
        return ServiceBand.CAUTION
    return ServiceBand.LOW


def resolve_dual_stage_result(base_high: bool, screening_high: bool) -> DualStagePolicyResult:
    service_band = resolve_dual_stage_band(base_high=base_high, screening_high=screening_high)
    return DualStagePolicyResult(
        service_band=service_band,
        service_band_label=get_service_band_label(service_band),
        service_band_percent=get_service_band_percent(service_band),
        legacy_risk_level=to_legacy_risk_level(service_band),
    )


def get_service_band_label(service_band: ServiceBand) -> str:
    return SERVICE_BAND_LABELS[service_band]


def get_service_band_percent(service_band: ServiceBand) -> int:
    return SERVICE_BAND_PERCENTS[service_band]


def to_legacy_risk_level(service_band: ServiceBand) -> str:
    """Return existing LOW/MEDIUM/HIGH risk level for DB/API compatibility only."""
    return LEGACY_RISK_LEVELS[service_band]
