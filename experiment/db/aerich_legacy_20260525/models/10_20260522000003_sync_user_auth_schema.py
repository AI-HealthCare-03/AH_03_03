from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" ALTER COLUMN "hashed_password" TYPE VARCHAR(255);
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "failed_login_count" INT NOT NULL DEFAULT 0;
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "locked_until" TIMESTAMPTZ;
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "last_login_at" TIMESTAMPTZ;
ALTER TABLE "users" DROP CONSTRAINT IF EXISTS "users_firebase_uid_key";
DROP INDEX IF EXISTS "users_firebase_uid_key";
DROP INDEX IF EXISTS "idx_users_auth_provider";
DROP INDEX IF EXISTS "idx_users_firebase_uid_unique";
ALTER TABLE "users" DROP COLUMN IF EXISTS "firebase_uid";
ALTER TABLE "users" DROP COLUMN IF EXISTS "auth_provider";
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "firebase_uid" VARCHAR(128);
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "auth_provider" VARCHAR(20) NOT NULL DEFAULT 'local';
CREATE UNIQUE INDEX IF NOT EXISTS "users_firebase_uid_key"
    ON "users" ("firebase_uid");
CREATE INDEX IF NOT EXISTS "idx_users_auth_provider"
    ON "users" ("auth_provider");
ALTER TABLE "users" DROP COLUMN IF EXISTS "last_login_at";
ALTER TABLE "users" DROP COLUMN IF EXISTS "locked_until";
ALTER TABLE "users" DROP COLUMN IF EXISTS "failed_login_count";
ALTER TABLE "users" ALTER COLUMN "hashed_password" TYPE VARCHAR(128);
"""
