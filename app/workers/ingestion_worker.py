import asyncio
import signal
import sys
import uuid
from app.core.logging import setup_logging, logger, correlation_id_ctx, business_id_ctx
from app.core.config import settings
from app.services.redis_service import redis_service
from app.services.ingestion_service import ingestion_service
from app.db.session import engine


class IngestionWorker:
    def __init__(self, embedded: bool = False) -> None:
        self.is_running = True
        self.embedded = embedded

    def stop(self) -> None:
        logger.info("Received termination signal, shutting down ingestion worker gracefully...")
        self.is_running = False

    def _register_signal_handlers(self) -> None:
        """Register signal handlers on the RUNNING event loop (inside async context)."""
        if self.embedded:
            return
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
        logger.info("Starting AI-CTO Telemetry Ingestion Background Worker (embedded=%s)...", self.embedded)

        # Register signal handlers on the active event loop if standalone
        self._register_signal_handlers()

        # Connect Redis if standalone
        if not self.embedded:
            await redis_service.connect()
        await ingestion_service.ensure_consumer_group()

        logger.info(
            "Worker listening on stream '%s' (group: '%s')",
            settings.REDIS_STREAM_KEY,
            settings.REDIS_CONSUMER_GROUP,
        )

        while self.is_running:
            cycle_id = f"ingest-{uuid.uuid4().hex[:8]}"
            correlation_id_ctx.set(cycle_id)
            business_id_ctx.set("")
            try:
                processed_count = await ingestion_service.process_batch(batch_size=100, block_ms=2000)
                if processed_count == 0:
                    # Small yield to event loop when queue is idle
                    await asyncio.sleep(0.2)
            except Exception as e:
                logger.error("Unexpected error in worker loop: %s", e, exc_info=True)
                await asyncio.sleep(2.0)

        # Cleanup on exit
        if not self.embedded:
            logger.info("Cleaning up resources on standalone ingestion worker exit...")
            await redis_service.disconnect()
            await engine.dispose()
        logger.info("Ingestion worker shutdown complete.")


def main():
    worker = IngestionWorker(embedded=False)
    try:
        asyncio.run(worker.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker interrupted, exiting.")


if __name__ == "__main__":
    main()
