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
from app.db.models.log_entry import LogEntry
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

    async def route_to_dlq(self, entry_id: str, fields: Dict[str, Any], reason: str) -> None:
        """
        Diverts a malformed or poison-pill message to the Dead-Letter Queue (DLQ).
        Prevents consumer worker crash loops while preserving unprocessable data for triage.
        """
        dlq_entry = {
            "original_entry_id": str(entry_id),
            "error_reason": str(reason)[:500],
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "worker_consumer": self.consumer_name,
            "raw_payload": json.dumps(fields) if isinstance(fields, dict) else str(fields),
        }
        if redis_service.redis:
            try:
                await redis_service.redis.xadd(settings.REDIS_DLQ_STREAM_KEY, dlq_entry)
                logger.warning(
                    "Quarantined poison-pill event %s to DLQ stream '%s' (Reason: %s)",
                    entry_id,
                    settings.REDIS_DLQ_STREAM_KEY,
                    reason,
                )
            except Exception as e:
                logger.error("Failed to write poison pill %s to DLQ: %s", entry_id, e)
        else:
            # In-memory DLQ buffer fallback
            dlq_buf = redis_service._memory_streams.setdefault(settings.REDIS_DLQ_STREAM_KEY, [])
            dlq_buf.append((entry_id, dlq_entry))
            logger.warning("Quarantined poison-pill event %s to in-memory DLQ buffer (Reason: %s)", entry_id, reason)

    async def _mirror_telemetry_errors_to_logs(self, records: List[Dict[str, Any]]) -> None:
        """
        Bridges client website errors and HTTP failures into log_entries and Redis Stream.
        Ensures uncaught JS errors, unhandled rejections, and HTTP 4xx/5xx errors from tracker.js
        immediately stream into the live Logs & Observability terminal.
        """
        log_entries_to_persist = []
        for r in records:
            ev_type = (r.get("event_type") or "").lower()
            status = r.get("status_code", 200)
            is_client_error = ev_type in ("js_error", "unhandled_rejection", "client_error") or status >= 400
            if not is_client_error:
                continue

            lvl = "error" if (status >= 500 or "error" in ev_type or "rejection" in ev_type) else "warn"
            meta = r.get("payload_metadata") or {}
            endpoint = r.get("endpoint") or "/"

            if ev_type == "js_error":
                msg = meta.get("message") or "Uncaught JavaScript exception in browser"
                fn = meta.get("filename") or ""
                line = meta.get("lineno") or 0
                content = f"[CLIENT-BROWSER] {msg} ({fn}:{line})" if fn else f"[CLIENT-BROWSER] {msg}"
            elif ev_type == "unhandled_rejection":
                reason = meta.get("reason") or "Unhandled Promise rejection in browser"
                content = f"[CLIENT-BROWSER] Unhandled Promise Rejection: {reason}"
            else:
                content = f"[HTTP-{status}] Ingress failure on {endpoint} (latency: {r.get('response_time_ms', 0)}ms)"

            entry_id = str(uuid.uuid4())
            ts = r.get("timestamp") or datetime.now(timezone.utc)
            parsed_fields = {
                "endpoint": endpoint,
                "status_code": status,
                "event_type": ev_type,
                "latency_ms": r.get("response_time_ms", 0.0),
                **meta,
            }

            stream_payload = {
                "id": entry_id,
                "business_id": str(r["business_id"]),
                "log_type": "browser" if "js" in ev_type or "client" in ev_type or "rejection" in ev_type else "application",
                "source": "client-browser" if "js" in ev_type or "client" in ev_type or "rejection" in ev_type else "api-gateway",
                "format": "json",
                "content": content[:10000],
                "parsed_fields": parsed_fields,
                "level": lvl,
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }

            # Add to Redis log stream for live-tail SSE
            await redis_service.add_to_stream("log:entries:stream", stream_payload)

            log_entries_to_persist.append(
                LogEntry(
                    id=uuid.UUID(entry_id),
                    business_id=r["business_id"],
                    log_type=stream_payload["log_type"],
                    source=stream_payload["source"],
                    format="json",
                    content=content[:10000],
                    parsed_fields=parsed_fields,
                    level=lvl,
                    timestamp=ts if isinstance(ts, datetime) else datetime.now(timezone.utc),
                    ingested_at=datetime.now(timezone.utc),
                )
            )

        if log_entries_to_persist and await is_db_available():
            try:
                async with AsyncSessionLocal() as session:
                    session.add_all(log_entries_to_persist)
                    await session.commit()
            except Exception as e:
                logger.debug("Failed saving mirrored LogEntries to DB: %s", e)

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
                try:
                    record = self._parse_stream_entry(entry_id, fields)
                    if record:
                        parsed_records.append(record)
                    else:
                        await self.route_to_dlq(entry_id, fields, reason="Missing or invalid business_id in stream entry")
                    ack_ids.append(entry_id)
                except Exception as parse_err:
                    await self.route_to_dlq(entry_id, fields, reason=f"Stream parsing exception: {parse_err}")
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

            # Mirror client-side error events into log_entries and Redis log stream
            if parsed_records:
                await self._mirror_telemetry_errors_to_logs(parsed_records)

            # Invalidate dashboard metrics cache so UI updates immediately
            biz_ids = {str(r["business_id"]) for r in parsed_records if "business_id" in r}
            for bid in biz_ids:
                await redis_service.invalidate_cache_pattern(f"dashboard:metrics:{bid}")

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
                try:
                    record = self._parse_stream_entry(entry_id, fields)
                    if record:
                        parsed_records.append(record)
                    else:
                        await self.route_to_dlq(entry_id, fields, reason="Missing or invalid business_id in stream entry")
                    ack_ids.append(entry_id)
                except Exception as parse_err:
                    logger.error("Failed to parse stream entry %s: %s", entry_id, parse_err)
                    await self.route_to_dlq(entry_id, fields, reason=f"Stream parsing exception: {parse_err}")
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

            # Mirror client-side error events into log_entries and Redis log stream
            if parsed_records:
                await self._mirror_telemetry_errors_to_logs(parsed_records)

            # Acknowledge messages in Redis
            if ack_ids:
                try:
                    await redis_service.redis.xack(self.stream_key, self.group_name, *ack_ids)
                except Exception as ack_err:
                    logger.error("Failed to XACK stream entries in %s: %s", self.stream_key, ack_err, exc_info=True)

            # Invalidate dashboard metrics cache so UI updates immediately
            biz_ids = {str(r["business_id"]) for r in parsed_records if "business_id" in r}
            for bid in biz_ids:
                await redis_service.invalidate_cache_pattern(f"dashboard:metrics:{bid}")

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
