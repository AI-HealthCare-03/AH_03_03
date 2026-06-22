from __future__ import annotations

import pytest

from ai_runtime.ml.inference.dual_stage_policy import (
    ServiceBand,
    apply_strict_screening_policy_v2,
    coerce_service_band,
    get_service_band_label,
    get_service_band_percent,
    resolve_dual_stage_band,
    resolve_dual_stage_result,
    to_legacy_risk_level,
)


@pytest.mark.parametrize(
    ("base_risk_level", "screening_high", "expected_band", "expected_label", "expected_percent"),
    [
        (ServiceBand.LOW, False, ServiceBand.LOW, "낮음", 25),
        (ServiceBand.LOW, True, ServiceBand.CAUTION, "주의", 65),
        (ServiceBand.ATTENTION, False, ServiceBand.ATTENTION, "관심 필요", 45),
        (ServiceBand.ATTENTION, True, ServiceBand.CAUTION, "주의", 65),
        (ServiceBand.CAUTION, False, ServiceBand.CAUTION, "주의", 65),
        (ServiceBand.CAUTION, True, ServiceBand.CAUTION, "주의", 65),
        (ServiceBand.HIGH_CAUTION, False, ServiceBand.HIGH_CAUTION, "높은 주의", 80),
        (ServiceBand.HIGH_CAUTION, True, ServiceBand.HIGH_CAUTION, "높은 주의", 80),
    ],
)
def test_resolve_dual_stage_band(
    base_risk_level: ServiceBand,
    screening_high: bool,
    expected_band: ServiceBand,
    expected_label: str,
    expected_percent: int,
) -> None:
    band = resolve_dual_stage_band(base_risk_level=base_risk_level, screening_high=screening_high)

    assert band == expected_band
    assert get_service_band_label(band) == expected_label
    assert get_service_band_percent(band) == expected_percent


def test_caution_screening_high_is_not_promoted_to_high_caution() -> None:
    band = resolve_dual_stage_band(base_risk_level=ServiceBand.CAUTION, screening_high=True)

    assert band == ServiceBand.CAUTION
    assert band != ServiceBand.HIGH_CAUTION
    assert get_service_band_label(band) == "주의"


def test_high_caution_base_result_is_preserved_even_when_screening_low() -> None:
    band = resolve_dual_stage_band(base_risk_level=ServiceBand.HIGH_CAUTION, screening_high=False)

    assert band == ServiceBand.HIGH_CAUTION
    assert get_service_band_label(band) == "높은 주의"


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
    result = resolve_dual_stage_result(base_risk_level=ServiceBand.LOW, screening_high=True)

    assert result.risk_level == "CAUTION"
    assert result.service_band == ServiceBand.CAUTION
    assert result.service_band_label == "주의"
    assert result.service_band_percent == 65
    assert result.legacy_risk_level == "CAUTION"


@pytest.mark.parametrize(
    ("base_risk_level", "screening_high", "strict_high", "expected_band"),
    [
        (ServiceBand.LOW, True, True, ServiceBand.CAUTION),
        (ServiceBand.ATTENTION, True, True, ServiceBand.CAUTION),
        (ServiceBand.CAUTION, False, True, ServiceBand.HIGH_CAUTION),
        (ServiceBand.CAUTION, True, False, ServiceBand.CAUTION),
        (ServiceBand.HIGH_CAUTION, False, False, ServiceBand.HIGH_CAUTION),
    ],
)
def test_apply_strict_screening_policy_v2_promotion_limits(
    base_risk_level: ServiceBand,
    screening_high: bool,
    strict_high: bool,
    expected_band: ServiceBand,
) -> None:
    result = apply_strict_screening_policy_v2(
        base_risk_level=base_risk_level,
        screening_high=screening_high,
        strict_high=strict_high,
    )

    assert result.service_band == expected_band
    assert result.risk_level == expected_band.value


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


@pytest.mark.parametrize(
    ("legacy_risk_level", "expected_band"),
    [
        ("MEDIUM", ServiceBand.CAUTION),
        ("HIGH", ServiceBand.HIGH_CAUTION),
    ],
)
def test_coerce_legacy_risk_level(legacy_risk_level: str, expected_band: ServiceBand) -> None:
    assert coerce_service_band(legacy_risk_level) == expected_band
