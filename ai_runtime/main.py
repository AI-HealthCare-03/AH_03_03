from __future__ import annotations

import asyncio
import logging
import signal

from ai_runtime.jobs.consumer import run_consumer_forever
from ai_runtime.jobs.scheduler import run_scheduler_forever
from app.core import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


async def _main() -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        logger.info("AI Worker shutdown requested")
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, request_shutdown)

    logger.info("AI Worker started. Redis Stream consumer is enabled.")
    tasks = [asyncio.create_task(run_consumer_forever(stop_event), name="redis-stream-consumer")]
    if config.SCHEDULER_ENABLED:
        tasks.append(asyncio.create_task(run_scheduler_forever(stop_event), name="notification-scheduler"))

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for task in done:
        exc = task.exception()
        if exc is not None:
            logger.error("AI Worker task failed", exc_info=(type(exc), exc, exc.__traceback__))
            stop_event.set()

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    logger.info("AI Worker stopped.")


if __name__ == "__main__":
    asyncio.run(_main())
