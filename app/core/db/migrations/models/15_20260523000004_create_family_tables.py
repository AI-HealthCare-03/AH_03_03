from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
CREATE TABLE IF NOT EXISTS "families" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(100) NOT NULL,
    "owner_user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "status" VARCHAR(7) NOT NULL DEFAULT 'ACTIVE',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_families_owner_user_id_3fc1d8" ON "families" ("owner_user_id");
CREATE INDEX IF NOT EXISTS "idx_families_status_8c76b4" ON "families" ("status");
CREATE TABLE IF NOT EXISTS "family_members" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "family_id" BIGINT NOT NULL REFERENCES "families" ("id") ON DELETE CASCADE,
    "user_id" BIGINT REFERENCES "users" ("id") ON DELETE SET NULL,
    "display_name" VARCHAR(100) NOT NULL,
    "phone_number" VARCHAR(30),
    "email" VARCHAR(255),
    "relation_type" VARCHAR(11) NOT NULL,
    "member_role" VARCHAR(9) NOT NULL DEFAULT 'MEMBER',
    "status" VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    "is_registered" BOOL NOT NULL DEFAULT TRUE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_family_members_family_id_934e12" ON "family_members" ("family_id");
CREATE INDEX IF NOT EXISTS "idx_family_members_user_id_f8e42a" ON "family_members" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_family_members_family_user_0d9a15" ON "family_members" ("family_id", "user_id");
CREATE INDEX IF NOT EXISTS "idx_family_members_family_status_45cd0e" ON "family_members" ("family_id", "status");
CREATE INDEX IF NOT EXISTS "idx_family_members_email_6f9a22" ON "family_members" ("email");
CREATE INDEX IF NOT EXISTS "idx_family_members_phone_4a1b36" ON "family_members" ("phone_number");
CREATE TABLE IF NOT EXISTS "family_invites" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "family_id" BIGINT NOT NULL REFERENCES "families" ("id") ON DELETE CASCADE,
    "inviter_user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "invitee_user_id" BIGINT REFERENCES "users" ("id") ON DELETE SET NULL,
    "invitee_email" VARCHAR(255),
    "invitee_phone" VARCHAR(30),
    "code_hash" VARCHAR(128) NOT NULL UNIQUE,
    "relation_type" VARCHAR(11) NOT NULL,
    "member_role" VARCHAR(9) NOT NULL DEFAULT 'MEMBER',
    "expires_at" TIMESTAMPTZ NOT NULL,
    "used_at" TIMESTAMPTZ,
    "status" VARCHAR(8) NOT NULL DEFAULT 'PENDING',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_family_invites_family_id_b51f08" ON "family_invites" ("family_id");
CREATE INDEX IF NOT EXISTS "idx_family_invites_inviter_f21460" ON "family_invites" ("inviter_user_id");
CREATE INDEX IF NOT EXISTS "idx_family_invites_invitee_user_342cd6" ON "family_invites" ("invitee_user_id");
CREATE INDEX IF NOT EXISTS "idx_family_invites_email_88f7bb" ON "family_invites" ("invitee_email");
CREATE INDEX IF NOT EXISTS "idx_family_invites_phone_564a70" ON "family_invites" ("invitee_phone");
CREATE INDEX IF NOT EXISTS "idx_family_invites_status_c94f2f" ON "family_invites" ("status");
CREATE INDEX IF NOT EXISTS "idx_family_invites_expires_07dace" ON "family_invites" ("expires_at");
CREATE TABLE IF NOT EXISTS "family_share_settings" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "family_id" BIGINT NOT NULL REFERENCES "families" ("id") ON DELETE CASCADE,
    "owner_user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "viewer_user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "share_health_records" BOOL NOT NULL DEFAULT FALSE,
    "share_analysis_results" BOOL NOT NULL DEFAULT FALSE,
    "share_diet_records" BOOL NOT NULL DEFAULT FALSE,
    "share_medications" BOOL NOT NULL DEFAULT FALSE,
    "share_challenges" BOOL NOT NULL DEFAULT FALSE,
    "share_exam_reports" BOOL NOT NULL DEFAULT FALSE,
    "receive_analysis_alerts" BOOL NOT NULL DEFAULT FALSE,
    "receive_abnormal_value_alerts" BOOL NOT NULL DEFAULT FALSE,
    "receive_medication_alerts" BOOL NOT NULL DEFAULT FALSE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_family_share_family_id_d847ac" ON "family_share_settings" ("family_id");
CREATE INDEX IF NOT EXISTS "idx_family_share_owner_89cf0d" ON "family_share_settings" ("owner_user_id");
CREATE INDEX IF NOT EXISTS "idx_family_share_viewer_95cf23" ON "family_share_settings" ("viewer_user_id");
CREATE INDEX IF NOT EXISTS "idx_family_share_pair_a4c92b" ON "family_share_settings" ("family_id", "owner_user_id", "viewer_user_id");
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
DROP TABLE IF EXISTS "family_share_settings";
DROP TABLE IF EXISTS "family_invites";
DROP TABLE IF EXISTS "family_members";
DROP TABLE IF EXISTS "families";
"""
