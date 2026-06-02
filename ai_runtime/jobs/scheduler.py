from __future__ import annotations

import asyncio
import logging

from tortoise import Tortoise

from app.core import config
from app.core.db.databases import TORTOISE_ORM
from app.services import notifications as notification_service

logger = logging.getLogger(__name__)


async def run_scheduler_forever(
    stop_event: asyncio.Event,
    *,
    enabled: bool | None = None,
    interval_seconds: int | None = None,
) -> None:
    scheduler_enabled = config.SCHEDULER_ENABLED if enabled is None else enabled
    if not scheduler_enabled:
        logger.info("Notification scheduler is disabled.")
        return

    interval = max(1, interval_seconds or config.SCHEDULER_INTERVAL_SECONDS)
    await _initialize_db()
    logger.info("Notification scheduler started.", extra={"interval_seconds": interval})

    while not stop_event.is_set():
        try:
            created_count = await notification_service.process_due_reminder_schedules()
            if created_count:
                logger.info("Notification scheduler created reminders.", extra={"created_count": created_count})
        except Exception:
            logger.exception("Notification scheduler tick failed.")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            continue

    logger.info("Notification scheduler stopped.")


async def _initialize_db() -> None:
    if Tortoise._inited:
        return
    await Tortoise.init(config=TORTOISE_ORM)
