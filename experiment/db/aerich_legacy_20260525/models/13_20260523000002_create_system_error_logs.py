from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
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
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
DROP TABLE IF EXISTS "system_error_logs";
"""
