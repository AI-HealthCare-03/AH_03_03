from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "firebase_uid" VARCHAR(128);
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "auth_provider" VARCHAR(20) NOT NULL DEFAULT 'local';
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "role" VARCHAR(20) NOT NULL DEFAULT 'USER';
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "nickname" VARCHAR(30);
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "profile_image_url" VARCHAR(500);
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "deactivated_at" TIMESTAMPTZ;
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "email_verified_at" TIMESTAMPTZ;
CREATE UNIQUE INDEX IF NOT EXISTS "idx_users_firebase_uid_unique"
    ON "users" ("firebase_uid");
CREATE INDEX IF NOT EXISTS "idx_users_auth_provider"
    ON "users" ("auth_provider");
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
DROP INDEX IF EXISTS "idx_users_auth_provider";
DROP INDEX IF EXISTS "idx_users_firebase_uid_unique";
ALTER TABLE "users" DROP COLUMN IF EXISTS "firebase_uid";
ALTER TABLE "users" DROP COLUMN IF EXISTS "auth_provider";
"""
