from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "occupation_code" VARCHAR(30);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "family_htn" VARCHAR(10);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "family_dm" VARCHAR(10);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "family_dyslipidemia" VARCHAR(10);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "smoking_status" VARCHAR(20);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "drinking_frequency" VARCHAR(30);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "drinking_amount" VARCHAR(30);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "walking_days_per_week" INT;
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "strength_days_per_week" INT;
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "strength_days_per_week";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "walking_days_per_week";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "drinking_amount";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "drinking_frequency";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "smoking_status";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "family_dyslipidemia";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "family_dm";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "family_htn";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "occupation_code";
"""
