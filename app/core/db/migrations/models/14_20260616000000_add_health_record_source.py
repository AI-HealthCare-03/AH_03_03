from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "source" VARCHAR(30) NOT NULL DEFAULT 'MANUAL';
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "health_records" DROP COLUMN IF EXISTS "source";
    """


MODELS_STATE = ""
