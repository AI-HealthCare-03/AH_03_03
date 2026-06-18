from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "user_settings"
        ALTER COLUMN "challenge_reminder_time" TYPE TIME
        USING "challenge_reminder_time"::TIME;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "user_settings"
        ALTER COLUMN "challenge_reminder_time" TYPE TIMETZ
        USING "challenge_reminder_time"::TIMETZ;
    """
