from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "challenges"
    ADD COLUMN IF NOT EXISTS "challenge_type" VARCHAR(10) NOT NULL DEFAULT 'GENERAL';
ALTER TABLE "challenges"
    ADD COLUMN IF NOT EXISTS "target_disease" VARCHAR(20) NOT NULL DEFAULT 'GENERAL';
ALTER TABLE "challenges"
    ADD COLUMN IF NOT EXISTS "difficulty" VARCHAR(10) NOT NULL DEFAULT 'NORMAL';
ALTER TABLE "challenges"
    ADD COLUMN IF NOT EXISTS "caution_message" TEXT;
ALTER TABLE "challenges"
    ADD COLUMN IF NOT EXISTS "contraindication_message" TEXT;
CREATE INDEX IF NOT EXISTS "idx_challenges_type_target_status_4e2a91"
    ON "challenges" ("challenge_type", "target_disease", "status");
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
DROP INDEX IF EXISTS "idx_challenges_type_target_status_4e2a91";
ALTER TABLE "challenges" DROP COLUMN IF EXISTS "contraindication_message";
ALTER TABLE "challenges" DROP COLUMN IF EXISTS "caution_message";
ALTER TABLE "challenges" DROP COLUMN IF EXISTS "difficulty";
ALTER TABLE "challenges" DROP COLUMN IF EXISTS "target_disease";
ALTER TABLE "challenges" DROP COLUMN IF EXISTS "challenge_type";
"""
