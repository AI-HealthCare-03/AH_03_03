"""Seed challenge master data from the team CSV.

This script is for local MVP/full-service demos only. It is not intended for
production or shared databases.
"""

import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")

from tortoise import Tortoise  # noqa: E402

from app.core.db.databases import TORTOISE_ORM  # noqa: E402
from app.models.challenges import (  # noqa: E402
    Challenge,
    ChallengeCategory,
    ChallengeDifficulty,
    ChallengeStatus,
    ChallengeTargetDisease,
    ChallengeType,
)

CSV_PATH = ROOT_DIR / "docs" / "data" / "challenges" / "challenge_new.csv"


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _parse_bool(value: str | None) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "y", "active", "활성"}


def _parse_int(value: str | None, default: int) -> int:
    try:
        parsed = int(_clean(value))
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _category(value: str | None) -> ChallengeCategory:
    raw = _clean(value).upper() or ChallengeCategory.HABIT.value
    if raw == "BLOOD_SUGAR":
        raw = ChallengeCategory.BLOOD_GLUCOSE.value
    return ChallengeCategory(raw)


def _challenge_type(value: str | None) -> ChallengeType:
    raw = _clean(value).upper() or ChallengeType.GENERAL.value
    return ChallengeType(raw)


def _target_disease(value: str | None) -> ChallengeTargetDisease:
    raw = _clean(value).upper() or ChallengeTargetDisease.GENERAL.value
    return ChallengeTargetDisease(raw)


def _difficulty(value: str | None) -> ChallengeDifficulty:
    raw = _clean(value).upper() or ChallengeDifficulty.NORMAL.value
    return ChallengeDifficulty(raw)


def _status(value: str | None) -> ChallengeStatus:
    return ChallengeStatus.ACTIVE if _parse_bool(value) else ChallengeStatus.INACTIVE


def _optional(value: str | None) -> str | None:
    cleaned = _clean(value)
    return cleaned or None


def _load_challenge_rows() -> list[dict[str, Any]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Challenge master CSV not found: {CSV_PATH}")

    rows: list[dict[str, Any]] = []
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader, start=2):
            title = _clean(row.get("title"))
            if not title:
                raise ValueError(f"Missing title at CSV line {index}")
            rows.append(
                {
                    "challenge_type": _challenge_type(row.get("challenge_type")),
                    "target_disease": _target_disease(row.get("target_disease")),
                    "category": _category(row.get("category")),
                    "title": title,
                    "description": _optional(row.get("description")),
                    "duration_days": _parse_int(row.get("duration_days"), 7),
                    "target_metric": _optional(row.get("target_metric")),
                    "target_value": _optional(row.get("target_value")),
                    "difficulty": _difficulty(row.get("difficulty")),
                    "caution_message": _optional(row.get("caution_message")),
                    "contraindication_message": _optional(row.get("contraindication_message")),
                    "status": _status(row.get("is_active")),
                }
            )
    return rows


async def seed_challenges() -> None:
    await Tortoise.init(config=TORTOISE_ORM)
    created_count = 0
    updated_count = 0
    deactivated_count = 0
    rows = _load_challenge_rows()
    master_keys = {(row["title"], row["category"]) for row in rows}

    for row in rows:
        existing = await Challenge.get_or_none(title=row["title"], category=row["category"])
        if existing is None:
            await Challenge.create(**row)
            created_count += 1
            continue

        changed_fields: list[str] = []
        for key, value in row.items():
            if getattr(existing, key) != value:
                setattr(existing, key, value)
                changed_fields.append(key)
        if changed_fields:
            await existing.save()
            updated_count += 1

    active_challenges = await Challenge.filter(status=ChallengeStatus.ACTIVE)
    for challenge in active_challenges:
        if (challenge.title, challenge.category) not in master_keys:
            challenge.status = ChallengeStatus.INACTIVE
            await challenge.save()
            deactivated_count += 1

    await Tortoise.close_connections()
    print("===== Team Challenge Master Seed =====")
    print(f"csv_path: {CSV_PATH}")
    print(f"created_count: {created_count}")
    print(f"updated_count: {updated_count}")
    print(f"deactivated_count: {deactivated_count}")


if __name__ == "__main__":
    asyncio.run(seed_challenges())
