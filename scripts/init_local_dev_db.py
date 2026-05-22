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
