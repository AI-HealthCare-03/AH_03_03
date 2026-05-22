"""Seed challenge master data for local MVP frontend demos.

This script is for local MVP testing only. It is not intended for production or
shared databases. Challenge rows are created idempotently by title.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")

from tortoise import Tortoise  # noqa: E402

from app.core.db.databases import TORTOISE_ORM  # noqa: E402
from app.models.challenges import Challenge, ChallengeCategory, ChallengeStatus  # noqa: E402


@dataclass(frozen=True)
class ChallengeSeed:
    title: str
    category: ChallengeCategory
    target_disease: str
    description: str
    target_metric: str
    target_value: str
    duration_days: int = 7


CHALLENGE_SEEDS = [
    ChallengeSeed(
        title="하루 20분 걷기",
        category=ChallengeCategory.EXERCISE,
        target_disease="COMMON",
        description="하루 20분 이상 가볍게 걷는 기본 활동 챌린지입니다.",
        target_metric="exercise_minutes",
        target_value="20",
    ),
    ChallengeSeed(
        title="식후 10분 산책",
        category=ChallengeCategory.BLOOD_GLUCOSE,
        target_disease="DIABETES",
        description="혈당 관리를 위해 식후 10분 산책을 기록합니다.",
        target_metric="post_meal_walk_minutes",
        target_value="10",
    ),
    ChallengeSeed(
        title="단 음료 대신 물 마시기",
        category=ChallengeCategory.HABIT,
        target_disease="DIABETES",
        description="단 음료를 줄이고 물을 선택하는 습관을 기록합니다.",
        target_metric="water_replacement_count",
        target_value="1",
    ),
    ChallengeSeed(
        title="하루 한 끼 저당 식단 실천",
        category=ChallengeCategory.DIET,
        target_disease="DIABETES",
        description="하루 한 끼는 당류와 정제 탄수화물을 줄인 식단으로 기록합니다.",
        target_metric="low_sugar_meal_count",
        target_value="1",
    ),
    ChallengeSeed(
        title="야식 줄이기",
        category=ChallengeCategory.DIET,
        target_disease="OBESITY",
        description="비만 관리와 수면 질 개선을 위해 야식을 줄입니다.",
        target_metric="late_night_snack_count",
        target_value="0",
    ),
    ChallengeSeed(
        title="7시간 수면 챌린지",
        category=ChallengeCategory.SLEEP,
        target_disease="COMMON",
        description="하루 7시간 이상 수면을 목표로 합니다.",
        target_metric="sleep_hours",
        target_value="7",
    ),
    ChallengeSeed(
        title="복약 시간 기록하기",
        category=ChallengeCategory.HABIT,
        target_disease="COMMON",
        description="매일 복약 시간을 확인하고 기록하는 습관을 만듭니다.",
        target_metric="medication_record_count",
        target_value="1",
    ),
    ChallengeSeed(
        title="아침 건강 지표 입력하기",
        category=ChallengeCategory.BLOOD_PRESSURE,
        target_disease="COMMON",
        description="아침 혈압 또는 건강 지표를 기록합니다.",
        target_metric="morning_health_record_count",
        target_value="1",
    ),
]


async def seed_challenges() -> None:
    await Tortoise.init(config=TORTOISE_ORM)
    created_count = 0
    skipped_count = 0
    for seed in CHALLENGE_SEEDS:
        existing = await Challenge.get_or_none(title=seed.title)
        if existing is not None:
            skipped_count += 1
            continue

        await Challenge.create(
            title=seed.title,
            description=seed.description,
            category=seed.category,
            target_metric=seed.target_metric,
            target_value=seed.target_value,
            duration_days=seed.duration_days,
            status=ChallengeStatus.ACTIVE,
        )
        created_count += 1

    await Tortoise.close_connections()
    print("===== MVP Challenge Seed =====")
    print(f"created_count: {created_count}")
    print(f"skipped_count: {skipped_count}")


if __name__ == "__main__":
    asyncio.run(seed_challenges())
