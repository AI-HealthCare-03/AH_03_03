from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
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
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
DROP TABLE IF EXISTS "sensitive_access_logs";
"""
