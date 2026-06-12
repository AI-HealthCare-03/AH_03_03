from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" DROP COLUMN IF EXISTS "last_login";
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "last_login" TIMESTAMPTZ;
"""
