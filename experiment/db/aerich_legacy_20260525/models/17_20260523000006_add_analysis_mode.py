from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "analysis_results"
    ADD COLUMN IF NOT EXISTS "analysis_mode" VARCHAR(9) NOT NULL DEFAULT 'BASIC';
CREATE INDEX IF NOT EXISTS "idx_analysis_results_mode_1a9f30"
    ON "analysis_results" ("analysis_mode");
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
DROP INDEX IF EXISTS "idx_analysis_results_mode_1a9f30";
ALTER TABLE "analysis_results" DROP COLUMN IF EXISTS "analysis_mode";
"""
