from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.dtos.health import HealthRecordCreateRequest
from app.services import health as health_service


@pytest.mark.asyncio
async def test_create_health_record_preserves_existing_precision_extension_fields(monkeypatch) -> None:
    created_payload: dict[str, object] = {}

    async def fake_get_latest_health_record_by_user(user_id: int):
        assert user_id == 10
        return SimpleNamespace(
            ast=30,
            alt=25,
            gamma_gtp=44,
            creatinine=Decimal("0.90"),
            egfr=Decimal("82"),
            hemoglobin=Decimal("13.20"),
        )

    async def fake_create_health_record(user_id: int, data: dict):
        assert user_id == 10
        created_payload.update(data)
        return SimpleNamespace(id=101, user_id=user_id, **data)

    monkeypatch.setattr(
        health_service.health_repository,
        "get_latest_health_record_by_user",
        fake_get_latest_health_record_by_user,
    )
    monkeypatch.setattr(health_service.health_repository, "create_health_record", fake_create_health_record)

    await health_service.create_health_record(
        10,
        HealthRecordCreateRequest(
            height_cm=Decimal("170"),
            weight_kg=Decimal("70"),
            measured_at=datetime.now(UTC),
            source="MANUAL",
        ),
    )

    assert created_payload["ast"] == 30
    assert created_payload["alt"] == 25
    assert created_payload["gamma_gtp"] == 44
    assert created_payload["creatinine"] == Decimal("0.90")
    assert created_payload["egfr"] == Decimal("82")
    assert created_payload["hemoglobin"] == Decimal("13.20")
    assert created_payload["source"] == "MANUAL"
    assert created_payload["bmi"] == Decimal("24.22")


@pytest.mark.asyncio
async def test_create_health_record_keeps_new_precision_extension_input(monkeypatch) -> None:
    created_payload: dict[str, object] = {}

    async def fake_get_latest_health_record_by_user(user_id: int):
        assert user_id == 10
        return SimpleNamespace(ast=30, alt=25)

    async def fake_create_health_record(user_id: int, data: dict):
        assert user_id == 10
        created_payload.update(data)
        return SimpleNamespace(id=102, user_id=user_id, **data)

    monkeypatch.setattr(
        health_service.health_repository,
        "get_latest_health_record_by_user",
        fake_get_latest_health_record_by_user,
    )
    monkeypatch.setattr(health_service.health_repository, "create_health_record", fake_create_health_record)

    await health_service.create_health_record(
        10,
        HealthRecordCreateRequest(
            ast=35,
            alt=31,
            measured_at=datetime.now(UTC),
            source="MANUAL",
        ),
    )

    assert created_payload["ast"] == 35
    assert created_payload["alt"] == 31
