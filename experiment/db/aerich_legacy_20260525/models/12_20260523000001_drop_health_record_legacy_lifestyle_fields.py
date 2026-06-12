from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "is_smoker";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "drinks_alcohol";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "exercise_days_per_week";
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "is_smoker" BOOL;
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "drinks_alcohol" BOOL;
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "exercise_days_per_week" INT;
"""
