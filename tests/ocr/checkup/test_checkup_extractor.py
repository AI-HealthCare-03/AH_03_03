from __future__ import annotations

from ai_runtime.ocr.checkup.extractor import parse_blood_pressure, parse_from_text_lines, parse_height_weight


def test_parse_height_weight_does_not_reuse_150cm_as_weight() -> None:
    height, weight = parse_height_weight([("신장 150cm 체중 55kg", 0.99)])

    assert height == 150
    assert weight == 55


def test_parse_height_weight_keeps_high_weight_candidate() -> None:
    height, weight = parse_height_weight([("신장 165cm 체중 120kg", 0.99)])

    assert height == 165
    assert weight == 120


def test_parse_blood_pressure_orders_systolic_and_diastolic() -> None:
    systolic, diastolic = parse_blood_pressure([("혈압 80/120 mmHg", 0.99)])

    assert systolic == 120
    assert diastolic == 80


def test_parse_from_text_lines_marks_low_confidence_hb() -> None:
    data, low_conf, _ = parse_from_text_lines([("혈색소 13.2 g/dL", 0.5)])

    assert data.hb == 13.2
    assert "hb" in low_conf


def test_parse_from_text_lines_does_not_mark_high_confidence_hb() -> None:
    data, low_conf, _ = parse_from_text_lines([("혈색소 13.2 g/dL", 0.95)])

    assert data.hb == 13.2
    assert "hb" not in low_conf
