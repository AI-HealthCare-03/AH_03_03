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
"""


async def init_local_dev_db() -> None:
    await Tortoise.init(config=TORTOISE_ORM)
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
