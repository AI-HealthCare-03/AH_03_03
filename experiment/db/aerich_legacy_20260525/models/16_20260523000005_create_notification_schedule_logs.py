from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
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


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
DROP TABLE IF EXISTS "notification_logs";
DROP TABLE IF EXISTS "reminder_schedules";
"""
