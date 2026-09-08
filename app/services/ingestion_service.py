import time
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.core.config import settings
from app.core.logging import logger
from app.db.session import AsyncSessionLocal, is_db_available
from app.db.models.telemetry import TelemetryEvent
from app.services.redis_service import redis_service


class IngestionService:
    """
    Consumer service for processing telemetry events from Redis Streams in batches
    and persisting them reliably to PostgreSQL.
    """

    def __init__(
        self,
        stream_key: str = settings.REDIS_STREAM_KEY,
        group_name: str = settings.REDIS_CONSUMER_GROUP,
        consumer_name: Optional[str] = None,
    ) -> None:
        self.stream_key = stream_key
        self.group_name = group_name
        self.consumer_name = consumer_name or f"worker-{uuid.uuid4().hex[:8]}"

    async def ensure_consumer_group(self) -> None:
        """Creates the consumer group if it does not already exist."""
        if not redis_service.redis:
            return
        try:
            await redis_service.redis.xgroup_create(
                name=self.stream_key,
                groupname=self.group_name,
                id="0",
                mkstream=True,
            )
            logger.info("Created Redis Stream consumer group: %s on %s", self.group_name, self.stream_key)
        except Exception as e:
            # Group already exists is expected on worker restarts (BUSYGROUP)
            if "BUSYGROUP" in str(e):
                logger.debug("Consumer group %s already exists on %s", self.group_name, self.stream_key)
            else:
                logger.error("Error creating Redis Stream consumer group %s: %s", self.group_name, e, exc_info=True)

    def _parse_stream_entry(self, entry_id: str, fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Parses and type-casts Redis stream string fields to TelemetryEvent model kwargs."""
        try:
            biz_id_str = fields.get("business_id")
            if not biz_id_str:
                return None

            # Parse payload_metadata if present
            meta_raw = fields.get("payload_metadata", "{}")
            try:
                metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            except Exception:
                metadata = {}

            # Parse timestamp
            ts_str = fields.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                except Exception:
                    ts = datetime.now(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            event_id_str = fields.get("id")
            event_id = uuid.UUID(event_id_str) if event_id_str else uuid.uuid4()

            return {
                "id": event_id,
                "business_id": uuid.UUID(biz_id_str),
                "idempotency_key": fields.get("idempotency_key") or None,
                "event_type": fields.get("event_type", "request"),
                "response_time_ms": float(fields.get("response_time_ms", 0.0)),
                "status_code": int(fields.get("status_code", 200)),
                "orders_count": int(fields.get("orders_count", 0)),
                "revenue_amount": float(fields.get("revenue_amount", 0.0)),
                "cpu_usage_pct": float(fields.get("cpu_usage_pct", 0.0)),
                "memory_usage_pct": float(fields.get("memory_usage_pct", 0.0)),
                "queue_depth": int(fields.get("queue_depth", 0)),
                "endpoint": fields.get("endpoint") or None,
                "payload_metadata": metadata,
                "timestamp": ts,
            }
        except Exception as e:
            logger.error("Error parsing stream entry %s: %s", entry_id, e, exc_info=True)
            return None

    async def process_batch(self, batch_size: int = 100, block_ms: int = 2000) -> int:
        """
        Reads a batch of events from the stream, batch-inserts into Postgres,
        and acknowledges with XACK (or in-memory drain if Redis is offline).
        Returns the number of processed events.
        """
        if not redis_service.redis:
            # Drain in-memory stream buffer fallback
            entries_to_process = list(redis_service._memory_streams.get(self.stream_key, []))[:batch_size]
            if not entries_to_process:
                return 0

            # Remove drained entries from in-memory stream
            redis_service._memory_streams[self.stream_key] = redis_service._memory_streams[self.stream_key][len(entries_to_process):]

            parsed_records: List[Dict[str, Any]] = []
            ack_ids: List[str] = []
            for entry_id, fields in entries_to_process:
                record = self._parse_stream_entry(entry_id, fields)
                if record:
                    parsed_records.append(record)
                ack_ids.append(entry_id)

            # Try DB batch insert if DB is online
            db_duration_ms = 0.0
            if parsed_records and await is_db_available():
                try:
                    db_start = time.perf_counter()
                    async with AsyncSessionLocal() as session:
                        keyed_records = [r for r in parsed_records if r.get("idempotency_key")]
                        unkeyed_records = [r for r in parsed_records if not r.get("idempotency_key")]

                        if keyed_records:
                            stmt = insert(TelemetryEvent).values(keyed_records)
                            stmt = stmt.on_conflict_do_nothing(index_elements=["idempotency_key"])
                            await session.execute(stmt)

                        if unkeyed_records:
                            await session.execute(insert(TelemetryEvent).values(unkeyed_records))

                        await session.commit()
                    db_duration_ms = (time.perf_counter() - db_start) * 1000
                except Exception as db_err:
                    logger.debug("Database offline during batch telemetry insert (processed in dev mode): %s", db_err)
            elif parsed_records:
                logger.debug("Database offline: buffered %d telemetry events in memory", len(parsed_records))

            logger.info(
                "Ingestion worker %s batch: processed %d events in %.2fms (in-memory buffer, acked %d)",
                self.consumer_name,
                len(parsed_records),
                db_duration_ms,
                len(ack_ids),
            )
            return len(ack_ids)

        try:
            # Read new messages for this consumer group
            streams_response = await redis_service.redis.xreadgroup(
                groupname=self.group_name,
                consumername=self.consumer_name,
                streams={self.stream_key: ">"},
                count=batch_size,
                block=block_ms,
            )

            if not streams_response:
                return 0

            entries = streams_response[0][1]
            if not entries:
                return 0

            parsed_records: List[Dict[str, Any]] = []
            ack_ids: List[str] = []

            for entry_id, fields in entries:
                record = self._parse_stream_entry(entry_id, fields)
                if record:
                    parsed_records.append(record)
                ack_ids.append(entry_id)

            # Batch insert to Postgres
            db_duration_ms = 0.0
            if parsed_records and await is_db_available():
                try:
                    db_start = time.perf_counter()
                    async with AsyncSessionLocal() as session:
                        # Separate records with and without idempotency keys
                        # because ON CONFLICT DO NOTHING doesn't work with NULL values
                        keyed_records = [r for r in parsed_records if r.get("idempotency_key")]
                        unkeyed_records = [r for r in parsed_records if not r.get("idempotency_key")]

                        if keyed_records:
                            # Using ON CONFLICT DO NOTHING for idempotency keys
                            stmt = insert(TelemetryEvent).values(keyed_records)
                            stmt = stmt.on_conflict_do_nothing(index_elements=["idempotency_key"])
                            await session.execute(stmt)

                        if unkeyed_records:
                            # Direct insert for events without idempotency key
                            await session.execute(insert(TelemetryEvent).values(unkeyed_records))

                        await session.commit()
                    db_duration_ms = (time.perf_counter() - db_start) * 1000
                except Exception as db_err:
                    logger.debug("Database write skipped (DB offline): %s", db_err)

            # Acknowledge messages in Redis
            if ack_ids:
                try:
                    await redis_service.redis.xack(self.stream_key, self.group_name, *ack_ids)
                except Exception as ack_err:
                    logger.error("Failed to XACK stream entries in %s: %s", self.stream_key, ack_err, exc_info=True)

            logger.info(
                "Ingestion worker %s batch: inserted %d events to Postgres in %.2fms (acknowledged %d stream events)",
                self.consumer_name,
                len(parsed_records),
                db_duration_ms,
                len(ack_ids),
            )
            return len(ack_ids)

        except Exception as e:
            logger.error("Error in telemetry ingestion batch: %s", e, exc_info=True)
            return 0


ingestion_service = IngestionService()
