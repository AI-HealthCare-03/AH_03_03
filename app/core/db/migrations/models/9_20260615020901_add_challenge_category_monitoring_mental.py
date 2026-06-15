from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "analysis_snapshots" ALTER COLUMN "input_payload" TYPE JSONB USING "input_payload"::JSONB;
        ALTER TABLE "analysis_snapshots" ALTER COLUMN "output_payload" TYPE JSONB USING "output_payload"::JSONB;
        ALTER TABLE "analysis_snapshots" ALTER COLUMN "model_payload" TYPE JSONB USING "model_payload"::JSONB;
        ALTER TABLE "analysis_snapshots" ALTER COLUMN "shap_payload" TYPE JSONB USING "shap_payload"::JSONB;
        ALTER TABLE "async_jobs" ALTER COLUMN "request_payload" TYPE JSONB USING "request_payload"::JSONB;
        ALTER TABLE "async_jobs" ALTER COLUMN "result_payload" TYPE JSONB USING "result_payload"::JSONB;
        COMMENT ON COLUMN "challenges"."category" IS 'EXERCISE: EXERCISE
DIET: DIET
SLEEP: SLEEP
MEDICATION: MEDICATION
WATER: WATER
BLOOD_PRESSURE: BLOOD_PRESSURE
BLOOD_GLUCOSE: BLOOD_GLUCOSE
WEIGHT: WEIGHT
HABIT: HABIT
MONITORING: MONITORING
MENTAL: MENTAL';
        ALTER TABLE "diet_photo_results" ALTER COLUMN "raw_output" TYPE JSONB USING "raw_output"::JSONB;
        ALTER TABLE "diet_photo_results" ALTER COLUMN "detected_foods" TYPE JSONB USING "detected_foods"::JSONB;
        ALTER TABLE "diet_photo_results" ALTER COLUMN "confidence_payload" TYPE JSONB USING "confidence_payload"::JSONB;
        ALTER TABLE "diet_records" ALTER COLUMN "detected_foods" TYPE JSONB USING "detected_foods"::JSONB;
        ALTER TABLE "diet_records" ALTER COLUMN "nutrition_summary" TYPE JSONB USING "nutrition_summary"::JSONB;
        ALTER TABLE "llm_generation_logs" ALTER COLUMN "input_summary" TYPE JSONB USING "input_summary"::JSONB;
        ALTER TABLE "rag_retrieval_logs" ALTER COLUMN "retrieved_chunk_ids" TYPE JSONB USING "retrieved_chunk_ids"::JSONB;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "async_jobs" ALTER COLUMN "request_payload" TYPE JSONB USING "request_payload"::JSONB;
        ALTER TABLE "async_jobs" ALTER COLUMN "result_payload" TYPE JSONB USING "result_payload"::JSONB;
        COMMENT ON COLUMN "challenges"."category" IS 'EXERCISE: EXERCISE
DIET: DIET
SLEEP: SLEEP
MEDICATION: MEDICATION
WATER: WATER
BLOOD_PRESSURE: BLOOD_PRESSURE
BLOOD_GLUCOSE: BLOOD_GLUCOSE
WEIGHT: WEIGHT
HABIT: HABIT';
        ALTER TABLE "diet_records" ALTER COLUMN "detected_foods" TYPE JSONB USING "detected_foods"::JSONB;
        ALTER TABLE "diet_records" ALTER COLUMN "nutrition_summary" TYPE JSONB USING "nutrition_summary"::JSONB;
        ALTER TABLE "diet_photo_results" ALTER COLUMN "raw_output" TYPE JSONB USING "raw_output"::JSONB;
        ALTER TABLE "diet_photo_results" ALTER COLUMN "detected_foods" TYPE JSONB USING "detected_foods"::JSONB;
        ALTER TABLE "diet_photo_results" ALTER COLUMN "confidence_payload" TYPE JSONB USING "confidence_payload"::JSONB;
        ALTER TABLE "rag_retrieval_logs" ALTER COLUMN "retrieved_chunk_ids" TYPE JSONB USING "retrieved_chunk_ids"::JSONB;
        ALTER TABLE "analysis_snapshots" ALTER COLUMN "input_payload" TYPE JSONB USING "input_payload"::JSONB;
        ALTER TABLE "analysis_snapshots" ALTER COLUMN "output_payload" TYPE JSONB USING "output_payload"::JSONB;
        ALTER TABLE "analysis_snapshots" ALTER COLUMN "model_payload" TYPE JSONB USING "model_payload"::JSONB;
        ALTER TABLE "analysis_snapshots" ALTER COLUMN "shap_payload" TYPE JSONB USING "shap_payload"::JSONB;
        ALTER TABLE "llm_generation_logs" ALTER COLUMN "input_summary" TYPE JSONB USING "input_summary"::JSONB;"""


MODELS_STATE = (
    "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"
    "gdTK6q2rf/4fF9GmLQ=="
)
