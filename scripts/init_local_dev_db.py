"""Initialize local development DB tables from current Tortoise models.

This script is for local MVP testing only.
It is not a replacement for Aerich migrations and must not be used for
production/shared databases.
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DB_HOST", "localhost")

from tortoise import Tortoise  # noqa: E402

from app.core.db.databases import TORTOISE_ORM  # noqa: E402

PRE_GENERATE_SAFE_ALTER_SQL = """
ALTER TABLE IF EXISTS "analysis_results"
    ADD COLUMN IF NOT EXISTS "analysis_mode" VARCHAR(9) NOT NULL DEFAULT 'BASIC';
ALTER TABLE IF EXISTS "challenges"
    ADD COLUMN IF NOT EXISTS "challenge_type" VARCHAR(10) NOT NULL DEFAULT 'GENERAL';
ALTER TABLE IF EXISTS "challenges"
    ADD COLUMN IF NOT EXISTS "target_disease" VARCHAR(20) NOT NULL DEFAULT 'GENERAL';
ALTER TABLE IF EXISTS "challenges"
    ADD COLUMN IF NOT EXISTS "difficulty" VARCHAR(10) NOT NULL DEFAULT 'NORMAL';
ALTER TABLE IF EXISTS "challenges"
    ADD COLUMN IF NOT EXISTS "caution_message" TEXT;
ALTER TABLE IF EXISTS "challenges"
    ADD COLUMN IF NOT EXISTS "contraindication_message" TEXT;
"""

LOCAL_SAFE_ALTER_SQL = """
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "occupation_code" VARCHAR(30);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "family_htn" VARCHAR(10);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "family_dm" VARCHAR(10);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "family_dyslipidemia" VARCHAR(10);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "smoking_status" VARCHAR(20);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "drinking_frequency" VARCHAR(30);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "drinking_amount" VARCHAR(30);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "walking_days_per_week" INT;
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "strength_days_per_week" INT;
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "is_smoker";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "drinks_alcohol";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "exercise_days_per_week";
ALTER TABLE "analysis_results" ADD COLUMN IF NOT EXISTS "analysis_mode" VARCHAR(9) NOT NULL DEFAULT 'BASIC';
CREATE INDEX IF NOT EXISTS "idx_analysis_results_mode_1a9f30" ON "analysis_results" ("analysis_mode");
ALTER TABLE "challenges" ADD COLUMN IF NOT EXISTS "challenge_type" VARCHAR(10) NOT NULL DEFAULT 'GENERAL';
ALTER TABLE "challenges" ADD COLUMN IF NOT EXISTS "target_disease" VARCHAR(20) NOT NULL DEFAULT 'GENERAL';
ALTER TABLE "challenges" ADD COLUMN IF NOT EXISTS "difficulty" VARCHAR(10) NOT NULL DEFAULT 'NORMAL';
ALTER TABLE "challenges" ADD COLUMN IF NOT EXISTS "caution_message" TEXT;
ALTER TABLE "challenges" ADD COLUMN IF NOT EXISTS "contraindication_message" TEXT;
CREATE INDEX IF NOT EXISTS "idx_challenges_type_target_status_4e2a91"
    ON "challenges" ("challenge_type", "target_disease", "status");
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "failed_login_count" INT NOT NULL DEFAULT 0;
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "locked_until" TIMESTAMPTZ;
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "last_login_at" TIMESTAMPTZ;
ALTER TABLE "users" ALTER COLUMN "hashed_password" TYPE VARCHAR(255);
ALTER TABLE "users" DROP CONSTRAINT IF EXISTS "users_firebase_uid_key";
DROP INDEX IF EXISTS "users_firebase_uid_key";
DROP INDEX IF EXISTS "idx_users_auth_provider";
DROP INDEX IF EXISTS "idx_users_firebase_uid_unique";
ALTER TABLE "users" DROP COLUMN IF EXISTS "firebase_uid";
ALTER TABLE "users" DROP COLUMN IF EXISTS "auth_provider";
ALTER TABLE "users" DROP COLUMN IF EXISTS "last_login";
CREATE TABLE IF NOT EXISTS "system_error_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "request_id" VARCHAR(64),
    "user_id" BIGINT,
    "method" VARCHAR(10) NOT NULL,
    "path" VARCHAR(500) NOT NULL,
    "status_code" INT NOT NULL,
    "error_type" VARCHAR(100) NOT NULL,
    "error_message" TEXT,
    "stack_trace" TEXT,
    "client_ip" VARCHAR(100),
    "user_agent" VARCHAR(500),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_system_err_request_9d4b31" ON "system_error_logs" ("request_id");
CREATE INDEX IF NOT EXISTS "idx_system_err_user_id_f2a8d0" ON "system_error_logs" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_system_err_status_b8c5e1" ON "system_error_logs" ("status_code");
CREATE INDEX IF NOT EXISTS "idx_system_err_created_32f0ca" ON "system_error_logs" ("created_at");
CREATE TABLE IF NOT EXISTS "sensitive_access_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "request_id" VARCHAR(64),
    "actor_user_id" BIGINT NOT NULL,
    "actor_role" VARCHAR(30),
    "target_user_id" BIGINT NOT NULL,
    "action_type" VARCHAR(30) NOT NULL,
    "resource_type" VARCHAR(50) NOT NULL,
    "resource_id" BIGINT,
    "access_reason" VARCHAR(255),
    "method" VARCHAR(10) NOT NULL,
    "path" VARCHAR(500) NOT NULL,
    "client_ip" VARCHAR(100),
    "user_agent" VARCHAR(500),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_sensitive_access_request_6b0f3c" ON "sensitive_access_logs" ("request_id");
CREATE INDEX IF NOT EXISTS "idx_sensitive_access_actor_1e8c7a" ON "sensitive_access_logs" ("actor_user_id");
CREATE INDEX IF NOT EXISTS "idx_sensitive_access_target_4c2d95" ON "sensitive_access_logs" ("target_user_id");
CREATE INDEX IF NOT EXISTS "idx_sensitive_access_resource_e9b53f" ON "sensitive_access_logs" ("resource_type");
CREATE INDEX IF NOT EXISTS "idx_sensitive_access_created_74f628" ON "sensitive_access_logs" ("created_at");
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
CREATE TABLE IF NOT EXISTS "reminder_schedules" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "reminder_type" VARCHAR(13) NOT NULL,
    "channel" VARCHAR(6) NOT NULL DEFAULT 'IN_APP',
    "title" VARCHAR(100) NOT NULL,
    "message" TEXT NOT NULL,
    "related_type" VARCHAR(50),
    "related_id" BIGINT,
    "schedule_time" VARCHAR(8),
    "cron_expression" VARCHAR(100),
    "timezone" VARCHAR(50) NOT NULL DEFAULT 'Asia/Seoul',
    "is_active" BOOL NOT NULL DEFAULT TRUE,
    "last_triggered_at" TIMESTAMPTZ,
    "next_trigger_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_reminder_schedules_user_id_4f3c81" ON "reminder_schedules" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_reminder_schedules_user_active_b75f0e" ON "reminder_schedules" ("user_id", "is_active");
CREATE INDEX IF NOT EXISTS "idx_reminder_schedules_type_2bf72d" ON "reminder_schedules" ("reminder_type");
CREATE INDEX IF NOT EXISTS "idx_reminder_schedules_channel_39df6a" ON "reminder_schedules" ("channel");
CREATE INDEX IF NOT EXISTS "idx_reminder_schedules_next_95f7c6" ON "reminder_schedules" ("next_trigger_at");
ALTER TABLE "reminder_schedules"
    ALTER COLUMN "schedule_time" TYPE VARCHAR(8)
    USING LEFT("schedule_time"::TEXT, 8);
CREATE TABLE IF NOT EXISTS "notification_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "notification_id" BIGINT REFERENCES "notifications" ("id") ON DELETE SET NULL,
    "reminder_schedule_id" BIGINT REFERENCES "reminder_schedules" ("id") ON DELETE SET NULL,
    "notification_type" VARCHAR(30) NOT NULL,
    "channel" VARCHAR(6) NOT NULL DEFAULT 'IN_APP',
    "title" VARCHAR(100) NOT NULL,
    "message_summary" VARCHAR(255),
    "related_type" VARCHAR(50),
    "related_id" BIGINT,
    "status" VARCHAR(8) NOT NULL DEFAULT 'PENDING',
    "provider" VARCHAR(50),
    "provider_message_id" VARCHAR(120),
    "error_code" VARCHAR(50),
    "error_message" VARCHAR(255),
    "sent_at" TIMESTAMPTZ,
    "failed_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_notification_logs_user_id_f0a6b9" ON "notification_logs" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_notification_logs_notification_2ace41" ON "notification_logs" ("notification_id");
CREATE INDEX IF NOT EXISTS "idx_notification_logs_schedule_c4ad10" ON "notification_logs" ("reminder_schedule_id");
CREATE INDEX IF NOT EXISTS "idx_notification_logs_status_9cde70" ON "notification_logs" ("status");
CREATE INDEX IF NOT EXISTS "idx_notification_logs_channel_841db2" ON "notification_logs" ("channel");
CREATE INDEX IF NOT EXISTS "idx_notification_logs_created_11b98d" ON "notification_logs" ("created_at");
"""


async def init_local_dev_db() -> None:
    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.get_connection("default").execute_script(PRE_GENERATE_SAFE_ALTER_SQL)
    await Tortoise.generate_schemas(safe=True)
    # Local-only safety net: generate_schemas(safe=True) creates missing tables
    # but does not add new columns to existing tables. Shared/production DBs
    # must use Aerich migrations instead.
    await Tortoise.get_connection("default").execute_script(LOCAL_SAFE_ALTER_SQL)
    await Tortoise.close_connections()
    print("===== Local Development DB Init =====")
    print("Tortoise schemas generated with safe=True.")
    print("Local-only safe ALTER statements applied for MVP schema drift.")
    print("This is for local MVP testing only, not production migration management.")


if __name__ == "__main__":
    asyncio.run(init_local_dev_db())
