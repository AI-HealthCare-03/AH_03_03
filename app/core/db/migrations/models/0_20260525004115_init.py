from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "aerich" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "version" VARCHAR(255) NOT NULL,
    "app" VARCHAR(100) NOT NULL,
    "content" JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS "challenges" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "title" VARCHAR(100) NOT NULL,
    "description" TEXT,
    "category" VARCHAR(20) NOT NULL,
    "challenge_type" VARCHAR(10) NOT NULL DEFAULT 'GENERAL',
    "target_disease" VARCHAR(20) NOT NULL DEFAULT 'GENERAL',
    "difficulty" VARCHAR(10) NOT NULL DEFAULT 'NORMAL',
    "target_metric" VARCHAR(100),
    "target_value" VARCHAR(100),
    "caution_message" TEXT,
    "contraindication_message" TEXT,
    "duration_days" INT NOT NULL,
    "status" VARCHAR(8) NOT NULL DEFAULT 'ACTIVE',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_challenges_categor_bbb2d4" ON "challenges" ("category", "status");
CREATE INDEX IF NOT EXISTS "idx_challenges_challen_ed2619" ON "challenges" ("challenge_type", "target_disease", "status");
COMMENT ON COLUMN "challenges"."category" IS 'EXERCISE: EXERCISE\nDIET: DIET\nSLEEP: SLEEP\nMEDICATION: MEDICATION\nWATER: WATER\nBLOOD_PRESSURE: BLOOD_PRESSURE\nBLOOD_GLUCOSE: BLOOD_GLUCOSE\nWEIGHT: WEIGHT\nHABIT: HABIT';
COMMENT ON COLUMN "challenges"."challenge_type" IS 'SPECIAL: SPECIAL\nCOMMON: COMMON\nGENERAL: GENERAL';
COMMENT ON COLUMN "challenges"."target_disease" IS 'HYPERTENSION: HYPERTENSION\nDIABETES: DIABETES\nDYSLIPIDEMIA: DYSLIPIDEMIA\nOBESITY: OBESITY\nCOMMON: COMMON\nGENERAL: GENERAL';
COMMENT ON COLUMN "challenges"."difficulty" IS 'EASY: EASY\nNORMAL: NORMAL\nMEDIUM: MEDIUM\nHARD: HARD';
COMMENT ON COLUMN "challenges"."status" IS 'ACTIVE: ACTIVE\nINACTIVE: INACTIVE';
CREATE TABLE IF NOT EXISTS "faqs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "category" VARCHAR(50) NOT NULL,
    "question" VARCHAR(255) NOT NULL,
    "answer" TEXT NOT NULL,
    "display_order" INT NOT NULL DEFAULT 0,
    "is_active" BOOL NOT NULL DEFAULT True,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_faqs_categor_e46f1a" ON "faqs" ("category");
CREATE INDEX IF NOT EXISTS "idx_faqs_is_acti_77bf19" ON "faqs" ("is_active", "display_order");
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
CREATE INDEX IF NOT EXISTS "idx_sensitive_a_request_8c1576" ON "sensitive_access_logs" ("request_id");
CREATE INDEX IF NOT EXISTS "idx_sensitive_a_actor_u_056cb9" ON "sensitive_access_logs" ("actor_user_id");
CREATE INDEX IF NOT EXISTS "idx_sensitive_a_target__cea338" ON "sensitive_access_logs" ("target_user_id");
CREATE INDEX IF NOT EXISTS "idx_sensitive_a_resourc_c9602c" ON "sensitive_access_logs" ("resource_type");
CREATE INDEX IF NOT EXISTS "idx_sensitive_a_created_46117a" ON "sensitive_access_logs" ("created_at");
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
CREATE INDEX IF NOT EXISTS "idx_system_erro_request_6bd51a" ON "system_error_logs" ("request_id");
CREATE INDEX IF NOT EXISTS "idx_system_erro_user_id_edc59a" ON "system_error_logs" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_system_erro_status__157418" ON "system_error_logs" ("status_code");
CREATE INDEX IF NOT EXISTS "idx_system_erro_created_17d285" ON "system_error_logs" ("created_at");
CREATE TABLE IF NOT EXISTS "rag_sources" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(100) NOT NULL,
    "organization" VARCHAR(100),
    "source_type" VARCHAR(30) NOT NULL,
    "base_url" VARCHAR(500),
    "description" TEXT,
    "is_active" BOOL NOT NULL DEFAULT True,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS "rag_documents" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "title" VARCHAR(255) NOT NULL,
    "disease_type" VARCHAR(12) NOT NULL,
    "document_url" VARCHAR(500),
    "published_at" DATE,
    "fetched_at" TIMESTAMPTZ,
    "version" VARCHAR(50),
    "is_active" BOOL NOT NULL DEFAULT True,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "source_id" BIGINT NOT NULL REFERENCES "rag_sources" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_rag_documen_source__53260a" ON "rag_documents" ("source_id");
CREATE INDEX IF NOT EXISTS "idx_rag_documen_disease_15bf6f" ON "rag_documents" ("disease_type", "is_active");
COMMENT ON COLUMN "rag_documents"."disease_type" IS 'DIABETES: DIABETES\nOBESITY: OBESITY\nDYSLIPIDEMIA: DYSLIPIDEMIA\nCOMMON: COMMON';
CREATE TABLE IF NOT EXISTS "rag_chunks" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "chunk_index" INT NOT NULL,
    "section_title" VARCHAR(255),
    "content" TEXT NOT NULL,
    "disease_type" VARCHAR(12) NOT NULL,
    "keywords" TEXT,
    "embedding_model" VARCHAR(100),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "document_id" BIGINT NOT NULL REFERENCES "rag_documents" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_rag_chunks_documen_b5c641" ON "rag_chunks" ("document_id", "chunk_index");
CREATE INDEX IF NOT EXISTS "idx_rag_chunks_disease_7d5404" ON "rag_chunks" ("disease_type");
COMMENT ON COLUMN "rag_chunks"."disease_type" IS 'DIABETES: DIABETES\nOBESITY: OBESITY\nDYSLIPIDEMIA: DYSLIPIDEMIA\nCOMMON: COMMON';
CREATE TABLE IF NOT EXISTS "users" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "login_id" VARCHAR(40) UNIQUE,
    "email" VARCHAR(40) NOT NULL,
    "hashed_password" VARCHAR(255) NOT NULL,
    "name" VARCHAR(20) NOT NULL,
    "nickname" VARCHAR(30),
    "gender" VARCHAR(6) NOT NULL,
    "birthday" DATE NOT NULL,
    "phone_number" VARCHAR(11) NOT NULL,
    "address" VARCHAR(255),
    "profile_image_url" VARCHAR(500),
    "role" VARCHAR(20) NOT NULL DEFAULT 'USER',
    "is_active" BOOL NOT NULL DEFAULT True,
    "is_admin" BOOL NOT NULL DEFAULT False,
    "last_login_at" TIMESTAMPTZ,
    "failed_login_count" INT NOT NULL DEFAULT 0,
    "locked_until" TIMESTAMPTZ,
    "deactivated_at" TIMESTAMPTZ,
    "email_verified_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON COLUMN "users"."gender" IS 'MALE: MALE\nFEMALE: FEMALE';
CREATE TABLE IF NOT EXISTS "user_challenges" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "status" VARCHAR(9) NOT NULL DEFAULT 'JOINED',
    "started_at" TIMESTAMPTZ NOT NULL,
    "completed_at" TIMESTAMPTZ,
    "canceled_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "challenge_id" BIGINT NOT NULL REFERENCES "challenges" ("id") ON DELETE CASCADE,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_user_challe_user_id_1d8bec" ON "user_challenges" ("user_id", "status");
CREATE INDEX IF NOT EXISTS "idx_user_challe_challen_3e63d3" ON "user_challenges" ("challenge_id");
COMMENT ON COLUMN "user_challenges"."status" IS 'JOINED: JOINED\nCOMPLETED: COMPLETED\nCANCELED: CANCELED';
CREATE TABLE IF NOT EXISTS "challenge_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "log_date" DATE NOT NULL,
    "is_completed" BOOL NOT NULL DEFAULT False,
    "memo" VARCHAR(255),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_challenge_id" BIGINT NOT NULL REFERENCES "user_challenges" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_challenge_l_user_ch_739c98" ON "challenge_logs" ("user_challenge_id", "log_date");
CREATE TABLE IF NOT EXISTS "diet_records" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "meal_type" VARCHAR(20),
    "meal_time" TIMESTAMPTZ,
    "description" TEXT,
    "image_path" VARCHAR(500),
    "detected_foods" JSONB,
    "nutrition_summary" JSONB,
    "diet_score" DOUBLE PRECISION,
    "diet_feedback" TEXT,
    "analysis_method" VARCHAR(30),
    "is_user_corrected" BOOL NOT NULL DEFAULT False,
    "memo" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_diet_record_user_id_debdfc" ON "diet_records" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_diet_record_user_id_52bc6f" ON "diet_records" ("user_id", "meal_time");
CREATE INDEX IF NOT EXISTS "idx_diet_record_analysi_43bdea" ON "diet_records" ("analysis_method");
CREATE TABLE IF NOT EXISTS "diet_photo_results" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "detected_foods" JSONB,
    "confidence_payload" JSONB,
    "raw_output" JSONB,
    "is_dummy" BOOL NOT NULL DEFAULT True,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "diet_record_id" BIGINT NOT NULL REFERENCES "diet_records" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_diet_photo__diet_re_28b91b" ON "diet_photo_results" ("diet_record_id");
CREATE TABLE IF NOT EXISTS "exam_reports" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "original_filename" VARCHAR(255) NOT NULL,
    "file_path" VARCHAR(500) NOT NULL,
    "exam_date" DATE,
    "ocr_status" VARCHAR(10) NOT NULL DEFAULT 'PENDING',
    "is_confirmed" BOOL NOT NULL DEFAULT False,
    "uploaded_at" TIMESTAMPTZ NOT NULL,
    "confirmed_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_exam_report_user_id_30240a" ON "exam_reports" ("user_id", "uploaded_at");
CREATE INDEX IF NOT EXISTS "idx_exam_report_user_id_8bfb92" ON "exam_reports" ("user_id", "exam_date");
CREATE INDEX IF NOT EXISTS "idx_exam_report_ocr_sta_0799b1" ON "exam_reports" ("ocr_status");
COMMENT ON COLUMN "exam_reports"."ocr_status" IS 'PENDING: PENDING\nPROCESSING: PROCESSING\nSUCCESS: SUCCESS\nFAILED: FAILED\nCONFIRMED: CONFIRMED';
CREATE TABLE IF NOT EXISTS "exam_measurements" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "measurement_key" VARCHAR(100) NOT NULL,
    "measurement_name" VARCHAR(100) NOT NULL,
    "value" VARCHAR(100),
    "unit" VARCHAR(30),
    "ocr_confidence" DECIMAL(5,4),
    "is_user_confirmed" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "exam_report_id" BIGINT NOT NULL REFERENCES "exam_reports" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_exam_measur_exam_re_404bf8" ON "exam_measurements" ("exam_report_id");
CREATE INDEX IF NOT EXISTS "idx_exam_measur_measure_a50831" ON "exam_measurements" ("measurement_key");
CREATE TABLE IF NOT EXISTS "inquiries" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "category" VARCHAR(50) NOT NULL,
    "title" VARCHAR(255) NOT NULL,
    "content" TEXT NOT NULL,
    "status" VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    "answer" TEXT,
    "answered_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_inquiries_user_id_6d574f" ON "inquiries" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_inquiries_status_00079c" ON "inquiries" ("status");
CREATE INDEX IF NOT EXISTS "idx_inquiries_categor_6b91b4" ON "inquiries" ("category");
CREATE TABLE IF NOT EXISTS "families" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(100) NOT NULL,
    "status" VARCHAR(7) NOT NULL DEFAULT 'ACTIVE',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "owner_user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_families_owner_u_337b09" ON "families" ("owner_user_id");
CREATE INDEX IF NOT EXISTS "idx_families_status_5c9be8" ON "families" ("status");
COMMENT ON COLUMN "families"."status" IS 'ACTIVE: ACTIVE\nREMOVED: REMOVED';
CREATE TABLE IF NOT EXISTS "family_invites" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "invitee_email" VARCHAR(255),
    "invitee_phone" VARCHAR(30),
    "code_hash" VARCHAR(128) NOT NULL UNIQUE,
    "relation_type" VARCHAR(11) NOT NULL,
    "member_role" VARCHAR(9) NOT NULL DEFAULT 'MEMBER',
    "expires_at" TIMESTAMPTZ NOT NULL,
    "used_at" TIMESTAMPTZ,
    "status" VARCHAR(8) NOT NULL DEFAULT 'PENDING',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "family_id" BIGINT NOT NULL REFERENCES "families" ("id") ON DELETE CASCADE,
    "invitee_user_id" BIGINT REFERENCES "users" ("id") ON DELETE SET NULL,
    "inviter_user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_family_invi_family__5e20b3" ON "family_invites" ("family_id");
CREATE INDEX IF NOT EXISTS "idx_family_invi_inviter_a60e9e" ON "family_invites" ("inviter_user_id");
CREATE INDEX IF NOT EXISTS "idx_family_invi_invitee_9688d8" ON "family_invites" ("invitee_user_id");
CREATE INDEX IF NOT EXISTS "idx_family_invi_invitee_ef00b4" ON "family_invites" ("invitee_email");
CREATE INDEX IF NOT EXISTS "idx_family_invi_invitee_44d221" ON "family_invites" ("invitee_phone");
CREATE INDEX IF NOT EXISTS "idx_family_invi_status_fef7c9" ON "family_invites" ("status");
CREATE INDEX IF NOT EXISTS "idx_family_invi_expires_74b951" ON "family_invites" ("expires_at");
COMMENT ON COLUMN "family_invites"."relation_type" IS 'SELF: SELF\nFATHER: FATHER\nMOTHER: MOTHER\nSPOUSE: SPOUSE\nCHILD: CHILD\nSIBLING: SIBLING\nGRANDPARENT: GRANDPARENT\nOTHER: OTHER';
COMMENT ON COLUMN "family_invites"."member_role" IS 'OWNER: OWNER\nMEMBER: MEMBER\nGUARDIAN: GUARDIAN\nDEPENDENT: DEPENDENT';
COMMENT ON COLUMN "family_invites"."status" IS 'PENDING: PENDING\nACCEPTED: ACCEPTED\nDECLINED: DECLINED\nEXPIRED: EXPIRED\nCANCELED: CANCELED';
CREATE TABLE IF NOT EXISTS "family_members" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "display_name" VARCHAR(100) NOT NULL,
    "phone_number" VARCHAR(30),
    "email" VARCHAR(255),
    "relation_type" VARCHAR(11) NOT NULL,
    "member_role" VARCHAR(9) NOT NULL DEFAULT 'MEMBER',
    "status" VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    "is_registered" BOOL NOT NULL DEFAULT True,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "family_id" BIGINT NOT NULL REFERENCES "families" ("id") ON DELETE CASCADE,
    "user_id" BIGINT REFERENCES "users" ("id") ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS "idx_family_memb_family__688f4c" ON "family_members" ("family_id");
CREATE INDEX IF NOT EXISTS "idx_family_memb_user_id_811df0" ON "family_members" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_family_memb_family__8144f3" ON "family_members" ("family_id", "user_id");
CREATE INDEX IF NOT EXISTS "idx_family_memb_family__6fc88b" ON "family_members" ("family_id", "status");
CREATE INDEX IF NOT EXISTS "idx_family_memb_email_3c0463" ON "family_members" ("email");
CREATE INDEX IF NOT EXISTS "idx_family_memb_phone_n_a84758" ON "family_members" ("phone_number");
COMMENT ON COLUMN "family_members"."relation_type" IS 'SELF: SELF\nFATHER: FATHER\nMOTHER: MOTHER\nSPOUSE: SPOUSE\nCHILD: CHILD\nSIBLING: SIBLING\nGRANDPARENT: GRANDPARENT\nOTHER: OTHER';
COMMENT ON COLUMN "family_members"."member_role" IS 'OWNER: OWNER\nMEMBER: MEMBER\nGUARDIAN: GUARDIAN\nDEPENDENT: DEPENDENT';
COMMENT ON COLUMN "family_members"."status" IS 'ACTIVE: ACTIVE\nINVITED: INVITED\nPENDING_UNREGISTERED: PENDING_UNREGISTERED\nREMOVED: REMOVED';
CREATE TABLE IF NOT EXISTS "family_share_settings" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "share_health_records" BOOL NOT NULL DEFAULT False,
    "share_analysis_results" BOOL NOT NULL DEFAULT False,
    "share_diet_records" BOOL NOT NULL DEFAULT False,
    "share_medications" BOOL NOT NULL DEFAULT False,
    "share_challenges" BOOL NOT NULL DEFAULT False,
    "share_exam_reports" BOOL NOT NULL DEFAULT False,
    "receive_analysis_alerts" BOOL NOT NULL DEFAULT False,
    "receive_abnormal_value_alerts" BOOL NOT NULL DEFAULT False,
    "receive_medication_alerts" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "family_id" BIGINT NOT NULL REFERENCES "families" ("id") ON DELETE CASCADE,
    "owner_user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "viewer_user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_family_shar_family__2d322e" ON "family_share_settings" ("family_id");
CREATE INDEX IF NOT EXISTS "idx_family_shar_owner_u_ccf864" ON "family_share_settings" ("owner_user_id");
CREATE INDEX IF NOT EXISTS "idx_family_shar_viewer__598763" ON "family_share_settings" ("viewer_user_id");
CREATE INDEX IF NOT EXISTS "idx_family_shar_family__4908da" ON "family_share_settings" ("family_id", "owner_user_id", "viewer_user_id");
CREATE TABLE IF NOT EXISTS "health_records" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "height_cm" DECIMAL(5,2),
    "weight_kg" DECIMAL(5,2),
    "waist_cm" DECIMAL(5,2),
    "bmi" DECIMAL(5,2),
    "systolic_bp" INT,
    "diastolic_bp" INT,
    "fasting_glucose" INT,
    "hba1c" DECIMAL(4,2),
    "total_cholesterol" INT,
    "ldl_cholesterol" INT,
    "hdl_cholesterol" INT,
    "triglyceride" INT,
    "has_diabetes" BOOL,
    "has_obesity" BOOL,
    "has_dyslipidemia" BOOL,
    "has_hypertension" BOOL,
    "occupation_code" VARCHAR(30),
    "family_htn" VARCHAR(10),
    "family_dm" VARCHAR(10),
    "family_dyslipidemia" VARCHAR(10),
    "smoking_status" VARCHAR(20),
    "drinking_frequency" VARCHAR(30),
    "drinking_amount" VARCHAR(30),
    "walking_days_per_week" INT,
    "strength_days_per_week" INT,
    "sleep_hours" DECIMAL(4,2),
    "measured_at" TIMESTAMPTZ NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_health_reco_user_id_ef3622" ON "health_records" ("user_id", "measured_at");
CREATE TABLE IF NOT EXISTS "analysis_results" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "async_job_id" BIGINT,
    "analysis_type" VARCHAR(12) NOT NULL,
    "analysis_mode" VARCHAR(9) NOT NULL DEFAULT 'BASIC',
    "risk_score" DECIMAL(6,5) NOT NULL,
    "risk_level" VARCHAR(6) NOT NULL,
    "summary" VARCHAR(255),
    "model_name" VARCHAR(100),
    "model_version" VARCHAR(50),
    "analyzed_at" TIMESTAMPTZ NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "health_record_id" BIGINT NOT NULL REFERENCES "health_records" ("id") ON DELETE CASCADE,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_analysis_re_user_id_e2aed4" ON "analysis_results" ("user_id", "analyzed_at");
CREATE INDEX IF NOT EXISTS "idx_analysis_re_health__472c30" ON "analysis_results" ("health_record_id", "analysis_type");
COMMENT ON COLUMN "analysis_results"."analysis_type" IS 'DIABETES: DIABETES\nOBESITY: OBESITY\nDYSLIPIDEMIA: DYSLIPIDEMIA\nHYPERTENSION: HYPERTENSION';
COMMENT ON COLUMN "analysis_results"."analysis_mode" IS 'BASIC: BASIC\nPRECISION: PRECISION';
COMMENT ON COLUMN "analysis_results"."risk_level" IS 'LOW: LOW\nMEDIUM: MEDIUM\nHIGH: HIGH';
CREATE TABLE IF NOT EXISTS "analysis_result_factors" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "factor_key" VARCHAR(100) NOT NULL,
    "factor_name" VARCHAR(100) NOT NULL,
    "factor_value" VARCHAR(100),
    "contribution_score" DECIMAL(10,6),
    "direction" VARCHAR(8) NOT NULL,
    "display_order" INT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "analysis_result_id" BIGINT NOT NULL REFERENCES "analysis_results" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_analysis_re_analysi_a3f488" ON "analysis_result_factors" ("analysis_result_id", "display_order");
COMMENT ON COLUMN "analysis_result_factors"."direction" IS 'POSITIVE: POSITIVE\nNEGATIVE: NEGATIVE\nNEUTRAL: NEUTRAL';
CREATE TABLE IF NOT EXISTS "analysis_snapshots" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "input_payload" JSONB NOT NULL,
    "output_payload" JSONB NOT NULL,
    "shap_payload" JSONB,
    "model_payload" JSONB,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "analysis_result_id" BIGINT NOT NULL REFERENCES "analysis_results" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "challenge_recommendations" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "reason" TEXT,
    "priority" INT NOT NULL DEFAULT 0,
    "is_selected" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "analysis_result_id" BIGINT NOT NULL REFERENCES "analysis_results" ("id") ON DELETE CASCADE,
    "challenge_id" BIGINT NOT NULL REFERENCES "challenges" ("id") ON DELETE CASCADE,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_challenge_r_user_id_b44d4b" ON "challenge_recommendations" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_challenge_r_analysi_988666" ON "challenge_recommendations" ("analysis_result_id");
CREATE INDEX IF NOT EXISTS "idx_challenge_r_challen_8a45be" ON "challenge_recommendations" ("challenge_id");
CREATE INDEX IF NOT EXISTS "idx_challenge_r_user_id_a42968" ON "challenge_recommendations" ("user_id", "is_selected");
CREATE TABLE IF NOT EXISTS "llm_generation_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
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
    "estimated_cost" DECIMAL(12,6),
    "status" VARCHAR(30) NOT NULL DEFAULT 'SUCCESS',
    "error_message" TEXT,
    "latency_ms" INT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT REFERENCES "users" ("id") ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS "idx_llm_generat_user_id_248834" ON "llm_generation_logs" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_llm_generat_target__a798dd" ON "llm_generation_logs" ("target_type", "target_id");
CREATE INDEX IF NOT EXISTS "idx_llm_generat_llm_tas_6a0c12" ON "llm_generation_logs" ("llm_task_type");
CREATE INDEX IF NOT EXISTS "idx_llm_generat_status_10cbf2" ON "llm_generation_logs" ("status");
CREATE TABLE IF NOT EXISTS "medications" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(100) NOT NULL,
    "medication_type" VARCHAR(20) NOT NULL,
    "dosage" VARCHAR(100),
    "frequency" VARCHAR(100),
    "reminder_time" TIMETZ,
    "is_active" BOOL NOT NULL DEFAULT True,
    "memo" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_medications_user_id_4b4ec3" ON "medications" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_medications_user_id_58f1a1" ON "medications" ("user_id", "is_active");
CREATE INDEX IF NOT EXISTS "idx_medications_medicat_e7b3ea" ON "medications" ("medication_type");
CREATE TABLE IF NOT EXISTS "medication_records" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "scheduled_at" TIMESTAMPTZ,
    "taken_at" TIMESTAMPTZ,
    "is_taken" BOOL NOT NULL DEFAULT False,
    "status" VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    "memo" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "medication_id" BIGINT NOT NULL REFERENCES "medications" ("id") ON DELETE CASCADE,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_medication__medicat_b3b0a9" ON "medication_records" ("medication_id");
CREATE INDEX IF NOT EXISTS "idx_medication__user_id_b659d3" ON "medication_records" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_medication__medicat_b441eb" ON "medication_records" ("medication_id", "scheduled_at");
CREATE INDEX IF NOT EXISTS "idx_medication__user_id_d7a94a" ON "medication_records" ("user_id", "scheduled_at");
CREATE TABLE IF NOT EXISTS "notifications" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "notification_type" VARCHAR(30) NOT NULL,
    "title" VARCHAR(100) NOT NULL,
    "message" TEXT NOT NULL,
    "is_read" BOOL NOT NULL DEFAULT False,
    "related_type" VARCHAR(50),
    "related_id" BIGINT,
    "read_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_notificatio_user_id_daa173" ON "notifications" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_notificatio_user_id_46dd57" ON "notifications" ("user_id", "is_read");
CREATE INDEX IF NOT EXISTS "idx_notificatio_notific_4c14ec" ON "notifications" ("notification_type");
COMMENT ON TABLE "notifications" IS 'User-facing notification inbox item.';
CREATE TABLE IF NOT EXISTS "reminder_schedules" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "reminder_type" VARCHAR(13) NOT NULL,
    "channel" VARCHAR(6) NOT NULL DEFAULT 'IN_APP',
    "title" VARCHAR(100) NOT NULL,
    "message" TEXT NOT NULL,
    "related_type" VARCHAR(50),
    "related_id" BIGINT,
    "schedule_time" VARCHAR(8),
    "cron_expression" VARCHAR(100),
    "timezone" VARCHAR(50) NOT NULL DEFAULT 'Asia/Seoul',
    "is_active" BOOL NOT NULL DEFAULT True,
    "last_triggered_at" TIMESTAMPTZ,
    "next_trigger_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_reminder_sc_user_id_187f61" ON "reminder_schedules" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_reminder_sc_user_id_f4745d" ON "reminder_schedules" ("user_id", "is_active");
CREATE INDEX IF NOT EXISTS "idx_reminder_sc_reminde_9db130" ON "reminder_schedules" ("reminder_type");
CREATE INDEX IF NOT EXISTS "idx_reminder_sc_channel_8b1d07" ON "reminder_schedules" ("channel");
CREATE INDEX IF NOT EXISTS "idx_reminder_sc_next_tr_29ced7" ON "reminder_schedules" ("next_trigger_at");
COMMENT ON COLUMN "reminder_schedules"."reminder_type" IS 'MEDICATION: MEDICATION\nCHALLENGE: CHALLENGE\nHEALTH_RECORD: HEALTH_RECORD\nFAMILY_ALERT: FAMILY_ALERT\nSYSTEM: SYSTEM';
COMMENT ON COLUMN "reminder_schedules"."channel" IS 'IN_APP: IN_APP\nEMAIL: EMAIL\nSMS: SMS\nPUSH: PUSH\nKAKAO: KAKAO';
COMMENT ON TABLE "reminder_schedules" IS 'User-owned reminder schedule definition.';
CREATE TABLE IF NOT EXISTS "notification_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
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
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "notification_id" BIGINT REFERENCES "notifications" ("id") ON DELETE SET NULL,
    "reminder_schedule_id" BIGINT REFERENCES "reminder_schedules" ("id") ON DELETE SET NULL,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_notificatio_user_id_a388bf" ON "notification_logs" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_notificatio_notific_eb6742" ON "notification_logs" ("notification_id");
CREATE INDEX IF NOT EXISTS "idx_notificatio_reminde_2672e7" ON "notification_logs" ("reminder_schedule_id");
CREATE INDEX IF NOT EXISTS "idx_notificatio_status_ee45b1" ON "notification_logs" ("status");
CREATE INDEX IF NOT EXISTS "idx_notificatio_channel_784ec4" ON "notification_logs" ("channel");
CREATE INDEX IF NOT EXISTS "idx_notificatio_created_744d92" ON "notification_logs" ("created_at");
COMMENT ON COLUMN "notification_logs"."channel" IS 'IN_APP: IN_APP\nEMAIL: EMAIL\nSMS: SMS\nPUSH: PUSH\nKAKAO: KAKAO';
COMMENT ON COLUMN "notification_logs"."status" IS 'PENDING: PENDING\nSENT: SENT\nFAILED: FAILED\nSKIPPED: SKIPPED\nCANCELED: CANCELED';
COMMENT ON TABLE "notification_logs" IS 'Notification delivery attempt history.';
CREATE TABLE IF NOT EXISTS "rag_retrieval_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "query_text" TEXT NOT NULL,
    "disease_type" VARCHAR(12),
    "retrieved_chunk_ids" JSONB,
    "top_k" INT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "analysis_result_id" BIGINT REFERENCES "analysis_results" ("id") ON DELETE SET NULL,
    "user_id" BIGINT REFERENCES "users" ("id") ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS "idx_rag_retriev_user_id_11081d" ON "rag_retrieval_logs" ("user_id", "created_at");
CREATE INDEX IF NOT EXISTS "idx_rag_retriev_analysi_73ad16" ON "rag_retrieval_logs" ("analysis_result_id");
COMMENT ON COLUMN "rag_retrieval_logs"."disease_type" IS 'DIABETES: DIABETES\nOBESITY: OBESITY\nDYSLIPIDEMIA: DYSLIPIDEMIA\nCOMMON: COMMON';
CREATE TABLE IF NOT EXISTS "user_settings" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "notification_enabled" BOOL NOT NULL DEFAULT True,
    "challenge_reminder_enabled" BOOL NOT NULL DEFAULT True,
    "challenge_reminder_time" TIMETZ,
    "medication_reminder_enabled" BOOL NOT NULL DEFAULT True,
    "diet_reminder_enabled" BOOL NOT NULL DEFAULT False,
    "marketing_agreed" BOOL NOT NULL DEFAULT False,
    "sensitive_data_agreed" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL UNIQUE REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "password_reset_tokens" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "token_hash" VARCHAR(128) NOT NULL,
    "is_used" BOOL NOT NULL DEFAULT False,
    "expires_at" TIMESTAMPTZ NOT NULL,
    "used_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "refresh_tokens" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "token_jti" VARCHAR(64) NOT NULL UNIQUE,
    "expires_at" TIMESTAMPTZ NOT NULL,
    "revoked_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "user_consents" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "terms_agreed" BOOL NOT NULL DEFAULT True,
    "privacy_agreed" BOOL NOT NULL DEFAULT True,
    "sensitive_data_agreed" BOOL NOT NULL DEFAULT False,
    "marketing_agreed" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" BIGINT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "verification_codes" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "email" VARCHAR(40) NOT NULL,
    "code_hash" VARCHAR(128) NOT NULL,
    "purpose" VARCHAR(30) NOT NULL DEFAULT 'EMAIL_VERIFICATION',
    "is_used" BOOL NOT NULL DEFAULT False,
    "expires_at" TIMESTAMPTZ NOT NULL,
    "verified_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """


MODELS_STATE = (
    "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"
    "KIit8="
)
