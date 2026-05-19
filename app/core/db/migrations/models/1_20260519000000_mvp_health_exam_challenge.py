from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "health_records" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "height_cm" NUMERIC(5,2),
    "weight_kg" NUMERIC(5,2),
    "waist_cm" NUMERIC(5,2),
    "bmi" NUMERIC(5,2),
    "systolic_bp" INT,
    "diastolic_bp" INT,
    "fasting_glucose" INT,
    "hba1c" NUMERIC(4,2),
    "total_cholesterol" INT,
    "ldl_cholesterol" INT,
    "hdl_cholesterol" INT,
    "triglyceride" INT,
    "has_diabetes" BOOL,
    "has_obesity" BOOL,
    "has_dyslipidemia" BOOL,
    "has_hypertension" BOOL,
    "is_smoker" BOOL,
    "drinks_alcohol" BOOL,
    "exercise_days_per_week" INT,
    "sleep_hours" NUMERIC(4,2),
    "measured_at" TIMESTAMPTZ NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_health_records_user_measured"
    ON "health_records" ("user_id", "measured_at");

ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "waist_cm" NUMERIC(5,2);
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "total_cholesterol" INT;
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "ldl_cholesterol" INT;
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "hdl_cholesterol" INT;
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "triglyceride" INT;
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "has_obesity" BOOL;
ALTER TABLE "health_records" ADD COLUMN IF NOT EXISTS "has_dyslipidemia" BOOL;

CREATE TABLE IF NOT EXISTS "analysis_results" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "health_record_id" BIGINT NOT NULL REFERENCES "health_records" ("id") ON DELETE CASCADE,
    "async_job_id" BIGINT,
    "analysis_type" VARCHAR(12) NOT NULL,
    "risk_score" NUMERIC(6,5) NOT NULL,
    "risk_level" VARCHAR(6) NOT NULL,
    "summary" VARCHAR(255),
    "model_name" VARCHAR(100),
    "model_version" VARCHAR(50),
    "analyzed_at" TIMESTAMPTZ NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_analysis_results_user_analyzed"
    ON "analysis_results" ("user_id", "analyzed_at");
CREATE INDEX IF NOT EXISTS "idx_analysis_results_record_type"
    ON "analysis_results" ("health_record_id", "analysis_type");
ALTER TABLE IF EXISTS "analysis_results" ALTER COLUMN "analysis_type" TYPE VARCHAR(12);

CREATE TABLE IF NOT EXISTS "analysis_result_factors" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "analysis_result_id" BIGINT NOT NULL REFERENCES "analysis_results" ("id") ON DELETE CASCADE,
    "factor_key" VARCHAR(100) NOT NULL,
    "factor_name" VARCHAR(100) NOT NULL,
    "factor_value" VARCHAR(100),
    "contribution_score" NUMERIC(10,6),
    "direction" VARCHAR(8) NOT NULL,
    "display_order" INT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_analysis_factors_result_order"
    ON "analysis_result_factors" ("analysis_result_id", "display_order");

CREATE TABLE IF NOT EXISTS "analysis_snapshots" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "analysis_result_id" BIGINT NOT NULL REFERENCES "analysis_results" ("id") ON DELETE CASCADE,
    "input_payload" JSONB NOT NULL,
    "output_payload" JSONB NOT NULL,
    "shap_payload" JSONB,
    "model_payload" JSONB,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "exam_reports" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "original_filename" VARCHAR(255) NOT NULL,
    "file_path" VARCHAR(500) NOT NULL,
    "exam_date" DATE,
    "ocr_status" VARCHAR(10) NOT NULL DEFAULT 'PENDING',
    "is_confirmed" BOOL NOT NULL DEFAULT FALSE,
    "uploaded_at" TIMESTAMPTZ NOT NULL,
    "confirmed_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_exam_reports_user_uploaded"
    ON "exam_reports" ("user_id", "uploaded_at");
CREATE INDEX IF NOT EXISTS "idx_exam_reports_user_exam_date"
    ON "exam_reports" ("user_id", "exam_date");
CREATE INDEX IF NOT EXISTS "idx_exam_reports_ocr_status"
    ON "exam_reports" ("ocr_status");

CREATE TABLE IF NOT EXISTS "exam_measurements" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "exam_report_id" BIGINT NOT NULL REFERENCES "exam_reports" ("id") ON DELETE CASCADE,
    "measurement_key" VARCHAR(100) NOT NULL,
    "measurement_name" VARCHAR(100) NOT NULL,
    "value" VARCHAR(100),
    "unit" VARCHAR(30),
    "ocr_confidence" NUMERIC(5,4),
    "is_user_confirmed" BOOL NOT NULL DEFAULT FALSE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_exam_measurements_report"
    ON "exam_measurements" ("exam_report_id");
CREATE INDEX IF NOT EXISTS "idx_exam_measurements_key"
    ON "exam_measurements" ("measurement_key");

CREATE TABLE IF NOT EXISTS "challenges" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "title" VARCHAR(100) NOT NULL,
    "description" TEXT,
    "category" VARCHAR(20) NOT NULL,
    "target_metric" VARCHAR(100),
    "target_value" VARCHAR(100),
    "duration_days" INT NOT NULL,
    "status" VARCHAR(8) NOT NULL DEFAULT 'ACTIVE',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_challenges_category_status"
    ON "challenges" ("category", "status");

CREATE TABLE IF NOT EXISTS "user_challenges" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "challenge_id" BIGINT NOT NULL REFERENCES "challenges" ("id") ON DELETE CASCADE,
    "status" VARCHAR(9) NOT NULL DEFAULT 'JOINED',
    "started_at" TIMESTAMPTZ NOT NULL,
    "completed_at" TIMESTAMPTZ,
    "canceled_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_user_challenges_user_status"
    ON "user_challenges" ("user_id", "status");
CREATE INDEX IF NOT EXISTS "idx_user_challenges_challenge"
    ON "user_challenges" ("challenge_id");

CREATE TABLE IF NOT EXISTS "challenge_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_challenge_id" BIGINT NOT NULL REFERENCES "user_challenges" ("id") ON DELETE CASCADE,
    "log_date" DATE NOT NULL,
    "is_completed" BOOL NOT NULL DEFAULT FALSE,
    "memo" VARCHAR(255),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_challenge_logs_challenge_date"
    ON "challenge_logs" ("user_challenge_id", "log_date");

CREATE TABLE IF NOT EXISTS "challenge_recommendations" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "analysis_result_id" BIGINT NOT NULL REFERENCES "analysis_results" ("id") ON DELETE CASCADE,
    "challenge_id" BIGINT NOT NULL REFERENCES "challenges" ("id") ON DELETE CASCADE,
    "reason" TEXT,
    "priority" INT,
    "is_selected" BOOL NOT NULL DEFAULT FALSE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_challenge_recommendations_user"
    ON "challenge_recommendations" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_challenge_recommendations_analysis"
    ON "challenge_recommendations" ("analysis_result_id");
CREATE INDEX IF NOT EXISTS "idx_challenge_recommendations_challenge"
    ON "challenge_recommendations" ("challenge_id");
CREATE INDEX IF NOT EXISTS "idx_challenge_recommendations_user_selected"
    ON "challenge_recommendations" ("user_id", "is_selected");
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "challenge_recommendations";
DROP TABLE IF EXISTS "challenge_logs";
DROP TABLE IF EXISTS "user_challenges";
DROP TABLE IF EXISTS "challenges";
DROP TABLE IF EXISTS "exam_measurements";
DROP TABLE IF EXISTS "exam_reports";
DROP TABLE IF EXISTS "analysis_snapshots";
DROP TABLE IF EXISTS "analysis_result_factors";
DROP TABLE IF EXISTS "analysis_results";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "has_dyslipidemia";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "has_obesity";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "triglyceride";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "hdl_cholesterol";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "ldl_cholesterol";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "total_cholesterol";
ALTER TABLE "health_records" DROP COLUMN IF EXISTS "waist_cm";
"""
