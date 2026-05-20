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


async def init_local_dev_db() -> None:
    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.generate_schemas(safe=True)
    await Tortoise.close_connections()
    print("===== Local Development DB Init =====")
    print("Tortoise schemas generated with safe=True.")
    print("This is for local MVP testing only, not production migration management.")


if __name__ == "__main__":
    asyncio.run(init_local_dev_db())
