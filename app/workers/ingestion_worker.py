import asyncio
import signal
import sys
from app.core.logging import setup_logging, logger
from app.core.config import settings
from app.services.redis_service import redis_service
from app.services.ingestion_service import ingestion_service
from app.db.session import engine


class IngestionWorker:
    def __init__(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        logger.info("Received termination signal, shutting down ingestion worker gracefully...")
        self.is_running = False

    def _register_signal_handlers(self) -> None:
        """Register signal handlers on the RUNNING event loop (inside async context)."""
        loop = asyncio.get_running_loop()
        try:
            # Unix: register SIGINT and SIGTERM on the active loop
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self.stop)
        except (NotImplementedError, AttributeError, ValueError):
            # Windows: add_signal_handler is not supported,
            # KeyboardInterrupt (Ctrl+C) is handled via try/except below
            logger.info("Signal handlers not supported on this platform, using KeyboardInterrupt fallback")

    async def run(self) -> None:
        setup_logging()
        logger.info("Starting AI-CTO Telemetry Ingestion Background Worker...")

        # Register signal handlers on the active event loop
        self._register_signal_handlers()

        # Connect Redis
        await redis_service.connect()
        await ingestion_service.ensure_consumer_group()

        logger.info(
            "Worker listening on stream '%s' (group: '%s')",
            settings.REDIS_STREAM_KEY,
            settings.REDIS_CONSUMER_GROUP,
        )

        while self.is_running:
            try:
                processed_count = await ingestion_service.process_batch(batch_size=100, block_ms=2000)
                if processed_count == 0:
                    # Small yield to event loop when queue is idle
                    await asyncio.sleep(0.2)
            except Exception as e:
                logger.error("Unexpected error in worker loop: %s", e, exc_info=True)
                await asyncio.sleep(2.0)

        # Cleanup on exit
        await redis_service.disconnect()
        await engine.dispose()
        logger.info("Ingestion worker shutdown complete.")


def main():
    worker = IngestionWorker()
    try:
        asyncio.run(worker.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker interrupted, exiting.")


if __name__ == "__main__":
    main()
