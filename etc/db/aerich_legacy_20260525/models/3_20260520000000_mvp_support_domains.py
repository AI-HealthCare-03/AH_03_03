from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
CREATE TABLE IF NOT EXISTS "notifications" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "notification_type" VARCHAR(30) NOT NULL,
    "title" VARCHAR(100) NOT NULL,
    "message" TEXT NOT NULL,
    "is_read" BOOL NOT NULL DEFAULT FALSE,
    "related_type" VARCHAR(50),
    "related_id" BIGINT,
    "read_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_notifications_user"
    ON "notifications" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_notifications_user_read"
    ON "notifications" ("user_id", "is_read");
CREATE INDEX IF NOT EXISTS "idx_notifications_type"
    ON "notifications" ("notification_type");

CREATE TABLE IF NOT EXISTS "user_settings" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL UNIQUE REFERENCES "users" ("id") ON DELETE CASCADE,
    "notification_enabled" BOOL NOT NULL DEFAULT TRUE,
    "challenge_reminder_enabled" BOOL NOT NULL DEFAULT TRUE,
    "challenge_reminder_time" TIME,
    "medication_reminder_enabled" BOOL NOT NULL DEFAULT TRUE,
    "diet_reminder_enabled" BOOL NOT NULL DEFAULT FALSE,
    "marketing_agreed" BOOL NOT NULL DEFAULT FALSE,
    "sensitive_data_agreed" BOOL NOT NULL DEFAULT FALSE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "faqs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "category" VARCHAR(50) NOT NULL,
    "question" VARCHAR(255) NOT NULL,
    "answer" TEXT NOT NULL,
    "display_order" INT NOT NULL DEFAULT 0,
    "is_active" BOOL NOT NULL DEFAULT TRUE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_faqs_category"
    ON "faqs" ("category");
CREATE INDEX IF NOT EXISTS "idx_faqs_active_order"
    ON "faqs" ("is_active", "display_order");

CREATE TABLE IF NOT EXISTS "inquiries" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "category" VARCHAR(50) NOT NULL,
    "title" VARCHAR(255) NOT NULL,
    "content" TEXT NOT NULL,
    "status" VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    "answer" TEXT,
    "answered_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_inquiries_user"
    ON "inquiries" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_inquiries_status"
    ON "inquiries" ("status");
CREATE INDEX IF NOT EXISTS "idx_inquiries_category"
    ON "inquiries" ("category");

CREATE TABLE IF NOT EXISTS "diet_records" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "meal_type" VARCHAR(20),
    "meal_time" TIMESTAMPTZ,
    "description" TEXT,
    "image_path" VARCHAR(500),
    "detected_foods" JSONB,
    "nutrition_summary" JSONB,
    "diet_score" DOUBLE PRECISION,
    "diet_feedback" TEXT,
    "analysis_method" VARCHAR(30),
    "is_user_corrected" BOOL NOT NULL DEFAULT FALSE,
    "memo" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_diet_records_user"
    ON "diet_records" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_diet_records_user_meal_time"
    ON "diet_records" ("user_id", "meal_time");
CREATE INDEX IF NOT EXISTS "idx_diet_records_analysis_method"
    ON "diet_records" ("analysis_method");

CREATE TABLE IF NOT EXISTS "diet_photo_results" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "diet_record_id" BIGINT NOT NULL REFERENCES "diet_records" ("id") ON DELETE CASCADE,
    "detected_foods" JSONB,
    "confidence_payload" JSONB,
    "raw_output" JSONB,
    "is_dummy" BOOL NOT NULL DEFAULT TRUE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_diet_photo_results_record"
    ON "diet_photo_results" ("diet_record_id");

CREATE TABLE IF NOT EXISTS "medications" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "name" VARCHAR(100) NOT NULL,
    "medication_type" VARCHAR(20) NOT NULL,
    "dosage" VARCHAR(100),
    "frequency" VARCHAR(100),
    "reminder_time" TIME,
    "is_active" BOOL NOT NULL DEFAULT TRUE,
    "memo" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_medications_user"
    ON "medications" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_medications_user_active"
    ON "medications" ("user_id", "is_active");
CREATE INDEX IF NOT EXISTS "idx_medications_type"
    ON "medications" ("medication_type");

CREATE TABLE IF NOT EXISTS "medication_records" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "medication_id" BIGINT NOT NULL REFERENCES "medications" ("id") ON DELETE CASCADE,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "scheduled_at" TIMESTAMPTZ,
    "taken_at" TIMESTAMPTZ,
    "is_taken" BOOL NOT NULL DEFAULT FALSE,
    "status" VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    "memo" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_medication_records_medication"
    ON "medication_records" ("medication_id");
CREATE INDEX IF NOT EXISTS "idx_medication_records_user"
    ON "medication_records" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_medication_records_medication_scheduled"
    ON "medication_records" ("medication_id", "scheduled_at");
CREATE INDEX IF NOT EXISTS "idx_medication_records_user_scheduled"
    ON "medication_records" ("user_id", "scheduled_at");

CREATE TABLE IF NOT EXISTS "llm_generation_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT REFERENCES "users" ("id") ON DELETE SET NULL,
    "target_type" VARCHAR(50),
    "target_id" BIGINT,
    "llm_task_type" VARCHAR(50) NOT NULL,
    "provider" VARCHAR(50),
    "model_name" VARCHAR(100),
    "prompt_version" VARCHAR(50),
    "input_summary" JSONB,
    "output_text" TEXT,
    "prompt_tokens" INT,
    "completion_tokens" INT,
    "total_tokens" INT,
    "estimated_cost" NUMERIC(12,6),
    "status" VARCHAR(30) NOT NULL DEFAULT 'SUCCESS',
    "error_message" TEXT,
    "latency_ms" INT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_llm_generation_logs_user"
    ON "llm_generation_logs" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_llm_generation_logs_target"
    ON "llm_generation_logs" ("target_type", "target_id");
CREATE INDEX IF NOT EXISTS "idx_llm_generation_logs_task"
    ON "llm_generation_logs" ("llm_task_type");
CREATE INDEX IF NOT EXISTS "idx_llm_generation_logs_status"
    ON "llm_generation_logs" ("status");
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
DROP TABLE IF EXISTS "llm_generation_logs";
DROP TABLE IF EXISTS "medication_records";
DROP TABLE IF EXISTS "medications";
DROP TABLE IF EXISTS "diet_photo_results";
DROP TABLE IF EXISTS "diet_records";
DROP TABLE IF EXISTS "inquiries";
DROP TABLE IF EXISTS "faqs";
DROP TABLE IF EXISTS "user_settings";
DROP TABLE IF EXISTS "notifications";
"""
