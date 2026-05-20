from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "login_id" VARCHAR(40);
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "nickname" VARCHAR(30);
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "address" VARCHAR(255);
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "profile_image_url" VARCHAR(500);
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "role" VARCHAR(20) NOT NULL DEFAULT 'USER';
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "deactivated_at" TIMESTAMPTZ;
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "email_verified_at" TIMESTAMPTZ;
CREATE UNIQUE INDEX IF NOT EXISTS "idx_users_login_id_unique"
    ON "users" ("login_id");

CREATE TABLE IF NOT EXISTS "verification_codes" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "email" VARCHAR(40) NOT NULL,
    "code_hash" VARCHAR(128) NOT NULL,
    "purpose" VARCHAR(30) NOT NULL DEFAULT 'EMAIL_VERIFICATION',
    "is_used" BOOL NOT NULL DEFAULT FALSE,
    "expires_at" TIMESTAMPTZ NOT NULL,
    "verified_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_verification_codes_email_purpose"
    ON "verification_codes" ("email", "purpose", "is_used");

CREATE TABLE IF NOT EXISTS "password_reset_tokens" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "token_hash" VARCHAR(128) NOT NULL,
    "is_used" BOOL NOT NULL DEFAULT FALSE,
    "expires_at" TIMESTAMPTZ NOT NULL,
    "used_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_password_reset_tokens_user"
    ON "password_reset_tokens" ("user_id", "is_used");
CREATE INDEX IF NOT EXISTS "idx_password_reset_tokens_hash"
    ON "password_reset_tokens" ("token_hash");

CREATE TABLE IF NOT EXISTS "refresh_tokens" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "token_jti" VARCHAR(64) NOT NULL UNIQUE,
    "expires_at" TIMESTAMPTZ NOT NULL,
    "revoked_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_refresh_tokens_user_revoked"
    ON "refresh_tokens" ("user_id", "revoked_at");

CREATE TABLE IF NOT EXISTS "user_consents" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "terms_agreed" BOOL NOT NULL DEFAULT TRUE,
    "privacy_agreed" BOOL NOT NULL DEFAULT TRUE,
    "sensitive_data_agreed" BOOL NOT NULL DEFAULT FALSE,
    "marketing_agreed" BOOL NOT NULL DEFAULT FALSE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_user_consents_user"
    ON "user_consents" ("user_id");
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
DROP TABLE IF EXISTS "user_consents";
DROP TABLE IF EXISTS "refresh_tokens";
DROP TABLE IF EXISTS "password_reset_tokens";
DROP TABLE IF EXISTS "verification_codes";
DROP INDEX IF EXISTS "idx_users_login_id_unique";
ALTER TABLE "users" DROP COLUMN IF EXISTS "email_verified_at";
ALTER TABLE "users" DROP COLUMN IF EXISTS "deactivated_at";
ALTER TABLE "users" DROP COLUMN IF EXISTS "role";
ALTER TABLE "users" DROP COLUMN IF EXISTS "profile_image_url";
ALTER TABLE "users" DROP COLUMN IF EXISTS "address";
ALTER TABLE "users" DROP COLUMN IF EXISTS "nickname";
ALTER TABLE "users" DROP COLUMN IF EXISTS "login_id";
"""
