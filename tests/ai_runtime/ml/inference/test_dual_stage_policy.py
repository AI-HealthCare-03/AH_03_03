from __future__ import annotations

import pytest

from ai_runtime.ml.inference.dual_stage_policy import (
    ServiceBand,
    get_service_band_label,
    get_service_band_percent,
    resolve_dual_stage_band,
    resolve_dual_stage_result,
    to_legacy_risk_level,
)


@pytest.mark.parametrize(
    ("base_high", "screening_high", "expected_band", "expected_label", "expected_percent"),
    [
        (False, False, ServiceBand.LOW, "낮음", 25),
        (True, False, ServiceBand.ATTENTION, "관심 필요", 45),
        (False, True, ServiceBand.CAUTION, "주의", 65),
        (True, True, ServiceBand.HIGH_CAUTION, "높은 주의", 80),
    ],
)
def test_resolve_dual_stage_band(
    base_high: bool,
    screening_high: bool,
    expected_band: ServiceBand,
    expected_label: str,
    expected_percent: int,
) -> None:
    band = resolve_dual_stage_band(base_high=base_high, screening_high=screening_high)

    assert band == expected_band
    assert get_service_band_label(band) == expected_label
    assert get_service_band_percent(band) == expected_percent


def test_base_high_screening_low_is_not_downgraded_to_low() -> None:
    band = resolve_dual_stage_band(base_high=True, screening_high=False)

    assert band == ServiceBand.ATTENTION
    assert band != ServiceBand.LOW
    assert get_service_band_label(band) == "관심 필요"


@pytest.mark.parametrize(
    ("service_band", "expected_label", "expected_percent"),
    [
        (ServiceBand.LOW, "낮음", 25),
        (ServiceBand.ATTENTION, "관심 필요", 45),
        (ServiceBand.CAUTION, "주의", 65),
        (ServiceBand.HIGH_CAUTION, "높은 주의", 80),
    ],
)
def test_service_band_display_values(
    service_band: ServiceBand,
    expected_label: str,
    expected_percent: int,
) -> None:
    assert get_service_band_label(service_band) == expected_label
    assert get_service_band_percent(service_band) == expected_percent


def test_resolve_dual_stage_result_includes_display_and_legacy_values() -> None:
    result = resolve_dual_stage_result(base_high=False, screening_high=True)

    assert result.risk_level == "CAUTION"
    assert result.service_band == ServiceBand.CAUTION
    assert result.service_band_label == "주의"
    assert result.service_band_percent == 65
    assert result.legacy_risk_level == "CAUTION"


@pytest.mark.parametrize(
    ("service_band", "expected_legacy_risk_level"),
    [
        (ServiceBand.LOW, "LOW"),
        (ServiceBand.ATTENTION, "ATTENTION"),
        (ServiceBand.CAUTION, "CAUTION"),
        (ServiceBand.HIGH_CAUTION, "HIGH_CAUTION"),
    ],
)
def test_to_legacy_risk_level(service_band: ServiceBand, expected_legacy_risk_level: str) -> None:
    assert to_legacy_risk_level(service_band) == expected_legacy_risk_level
