from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" ALTER COLUMN "hashed_password" TYPE VARCHAR(255);
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" ALTER COLUMN "hashed_password" TYPE VARCHAR(128);
"""
