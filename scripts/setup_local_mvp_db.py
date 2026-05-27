"""Set up a local MVP demo database.

This script is for local frontend/backend MVP testing only. It generates tables
from current Tortoise models and seeds demo data. Production/shared databases
must use the agreed Aerich migration workflow instead.
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")

from scripts.init_local_dev_db import init_local_dev_db  # noqa: E402
from scripts.seed_demo_users import seed_demo_users  # noqa: E402
from scripts.seed_mvp_challenges import seed_challenges  # noqa: E402
from scripts.seed_mvp_faqs import seed_faqs  # noqa: E402


async def setup_local_mvp_db() -> None:
    print("===== Local MVP DB Setup =====")
    print("This setup is for local MVP testing only. Do not use it for production/shared DBs.")
    await init_local_dev_db()
    await seed_challenges()
    await seed_faqs()
    await seed_demo_users()
    print("===== Local MVP DB Setup Complete =====")
    print("demo@example.com / Demo1234!")
    print("demo_high@example.com / Demo1234!")
    print("admin@example.com / Demo1234! (SUPER_ADMIN)")
    print("monitor@example.com / Demo1234! (MONITOR)")


if __name__ == "__main__":
    asyncio.run(setup_local_mvp_db())
