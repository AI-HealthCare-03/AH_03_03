from __future__ import annotations

import asyncio
import logging
import signal

from ai_runtime.jobs.consumer import run_consumer_forever

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

    logger.info("AI Worker started. Redis Stream DEMO_ECHO consumer is enabled.")
    await run_consumer_forever(stop_event)
    logger.info("AI Worker stopped.")


if __name__ == "__main__":
    asyncio.run(_main())
