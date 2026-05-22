from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "failed_login_count" INT NOT NULL DEFAULT 0;
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "locked_until" TIMESTAMPTZ;
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "last_login_at" TIMESTAMPTZ;
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" DROP COLUMN IF EXISTS "last_login_at";
ALTER TABLE "users" DROP COLUMN IF EXISTS "locked_until";
ALTER TABLE "users" DROP COLUMN IF EXISTS "failed_login_count";
"""
