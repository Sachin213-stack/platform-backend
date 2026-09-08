import asyncio
import signal
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select, delete, func, desc

from app.core.config import settings
from app.core.logging import (
    setup_logging,
    logger,
    correlation_id_ctx,
    business_id_ctx,
    set_correlation_id,
    set_business_id,
)
from app.db.session import AsyncSessionLocal, engine, is_db_available
from app.db.models.business import Business
from app.db.models.telemetry import TelemetryEvent
from app.db.models.ml import Anomaly, Forecast
from app.services.redis_service import redis_service


async def detect_anomalies_for_business(business_id: uuid.UUID) -> int:
    """
    Runs scikit-learn IsolationForest anomaly detection on recent telemetry
    for a specific business tenant. Returns count of new anomalies detected.
    """
    if not await is_db_available():
        logger.debug("Database offline: skipping ML anomaly detection for tenant %s", business_id)
        return 0

    now = datetime.now(timezone.utc)
    lookback = now - timedelta(hours=2)

    async with AsyncSessionLocal() as session:
        # Fetch recent telemetry data
        stmt = (
            select(TelemetryEvent)
            .where(
                TelemetryEvent.business_id == business_id,
                TelemetryEvent.timestamp >= lookback,
            )
            .order_by(desc(TelemetryEvent.timestamp))
            .limit(300)
        )
        result = await session.execute(stmt)
        events = result.scalars().all()

        if len(events) < 10:
            logger.debug(
                "Insufficient telemetry records (%d < 10) for ML anomaly detection on tenant %s",
                len(events),
                business_id,
            )
            return 0

        # Extract features for anomaly detection: [response_time_ms, cpu_usage_pct, memory_usage_pct]
        features = []
        valid_events = []
        for e in events:
            rt = e.response_time_ms or 0.0
            cpu = e.cpu_usage_pct or 0.0
            mem = e.memory_usage_pct or 0.0
            if rt > 0:
                features.append([rt, cpu, mem])
                valid_events.append(e)

        if len(features) < 10:
            return 0

        X = np.array(features)
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(X)

        avg_latency = float(np.mean(X[:, 0]))
        anomalies_created = 0

        for idx, pred in enumerate(preds):
            if pred == -1:  # Outlier detected
                event = valid_events[idx]
                actual_val = round(float(event.response_time_ms or 0.0), 1)

                # Skip if actual value is below average (only flag high latency)
                if actual_val <= avg_latency * 1.25:
                    continue

                severity = "critical" if actual_val > avg_latency * 2.5 else "high" if actual_val > avg_latency * 1.75 else "medium"
                
                anomaly = Anomaly(
                    business_id=business_id,
                    metric_name="response_time",
                    severity=severity,
                    expected_value=round(avg_latency, 1),
                    actual_value=actual_val,
                    confidence_score=0.92,
                    description=f"Automated IsolationForest detected response time outlier ({actual_val}ms vs baseline {round(avg_latency, 1)}ms) on endpoint {event.endpoint or 'unknown'}",
                    is_resolved=False,
                    detected_at=event.timestamp,
                )
                session.add(anomaly)
                anomalies_created += 1

        if anomalies_created > 0:
            await session.commit()
            # Invalidate cached dashboard metrics
            await redis_service.set_cache(f"dashboard:metrics:{business_id}", None, ttl_seconds=1)
            logger.info("Saved %d new anomalies for tenant %s", anomalies_created, business_id)

        return anomalies_created


async def generate_forecast_for_business(business_id: uuid.UUID) -> Optional[Forecast]:
    """
    Fits a linear extrapolation model on hourly request volumes to generate
    traffic forecast curve for the next 24 hours.
    """
    if not await is_db_available():
        logger.debug("Database offline: skipping ML traffic forecast for tenant %s", business_id)
        return None

    now = datetime.now(timezone.utc)
    lookback = now - timedelta(days=3)

    async with AsyncSessionLocal() as session:
        stmt = (
            select(
                func.date_trunc("hour", TelemetryEvent.timestamp).label("hour"),
                func.count(TelemetryEvent.id).label("req_count"),
            )
            .where(
                TelemetryEvent.business_id == business_id,
                TelemetryEvent.timestamp >= lookback,
            )
            .group_by("hour")
            .order_by("hour")
        )
        result = await session.execute(stmt)
        rows = result.all()

        if len(rows) < 6:
            logger.debug(
                "Insufficient historical datapoints (%d < 6) for forecasting on tenant %s",
                len(rows),
                business_id,
            )
            return None

        # Build feature matrix: hours since start
        start_time = rows[0].hour
        X = np.array([[(r.hour - start_time).total_seconds() / 3600.0] for r in rows])
        y = np.array([float(r.req_count) for r in rows])

        model = LinearRegression()
        model.fit(X, y)

        # Forecast next 24 hourly steps
        last_hour_idx = float(X[-1][0])
        future_hours = np.array([[last_hour_idx + i] for i in range(1, 25)])
        yhat = model.predict(future_hours)

        curve = {
            "timestamps": [(now + timedelta(hours=i)).isoformat() for i in range(1, 25)],
            "yhat": [max(0.0, round(float(val), 1)) for val in yhat],
            "yhat_lower": [max(0.0, round(float(val) * 0.85, 1)) for val in yhat],
            "yhat_upper": [round(float(val) * 1.15, 1) for val in yhat],
        }

        # Calculate potential crash risk if growth rate is alarming
        slope = float(model.coef_[0])
        crash_risk = min(100.0, max(0.0, round(slope * 4.5, 1)))

        forecast = Forecast(
            business_id=business_id,
            metric_name="traffic_24h",
            forecast_horizon="24h",
            crash_risk_pct=crash_risk,
            forecast_curve=curve,
            generated_at=now,
        )
        session.add(forecast)
        await session.commit()
        return forecast


async def enforce_retention_for_business(business: Business) -> int:
    """Deletes telemetry events that exceed the business tenant's retention window."""
    if not await is_db_available():
        logger.debug("Database offline: skipping telemetry retention enforcement for tenant %s", business.id)
        return 0

    retention_days = business.retention_days or 30
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    async with AsyncSessionLocal() as session:
        stmt = (
            delete(TelemetryEvent)
            .where(
                TelemetryEvent.business_id == business.id,
                TelemetryEvent.timestamp < cutoff,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        deleted_count = result.rowcount or 0
        return deleted_count


async def run_anomaly_detection_job() -> None:
    """Scheduled job executing anomaly detection across all active tenants."""
    job_name = "anomaly_detection"
    lock_name = f"ml_job:{job_name}"

    acquired = await redis_service.acquire_lock(lock_name, ttl_seconds=50)
    if not acquired:
        logger.debug("Skipping %s job run: distributed lock already held by another worker", job_name)
        return

    try:
        if await is_db_available():
            try:
                async with AsyncSessionLocal() as session:
                    businesses = (await session.execute(select(Business))).scalars().all()
            except Exception as db_err:
                logger.warning("Could not query businesses for %s job: %s", job_name, db_err)
                businesses = []
        else:
            if settings.ENVIRONMENT == "development":
                businesses = [
                    Business(
                        id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                        name="Development Retail Tenant",
                        slug="dev-retail",
                        plan_tier="starter",
                        retention_days=30,
                    )
                ]
            else:
                businesses = []

        if not businesses:
            logger.debug("No businesses found for %s job", job_name)
            return

        for biz in businesses:
            cid = f"job-ml-anomaly-{uuid.uuid4().hex[:8]}"
            set_correlation_id(cid)
            set_business_id(str(biz.id))

            start_time = time.perf_counter()
            logger.info("Starting ML %s job for tenant %s", job_name, biz.id)

            anomalies_found = 0
            try:
                anomalies_found = await detect_anomalies_for_business(biz.id)
            except Exception as e:
                logger.warning(
                    "Model execution failed for %s on tenant %s: %s",
                    job_name,
                    biz.id,
                    e,
                )
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "Completed ML %s job for tenant %s in %.2fms (found %d anomalies)",
                    job_name,
                    biz.id,
                    duration_ms,
                    anomalies_found,
                )
    finally:
        await redis_service.release_lock(lock_name)


async def run_forecasting_job() -> None:
    """Scheduled job executing traffic and capacity forecasting across all active tenants."""
    job_name = "forecasting"
    lock_name = f"ml_job:{job_name}"

    acquired = await redis_service.acquire_lock(lock_name, ttl_seconds=290)
    if not acquired:
        logger.debug("Skipping %s job run: distributed lock already held by another worker", job_name)
        return

    try:
        if await is_db_available():
            try:
                async with AsyncSessionLocal() as session:
                    businesses = (await session.execute(select(Business))).scalars().all()
            except Exception as db_err:
                logger.warning("Could not query businesses for %s job: %s", job_name, db_err)
                businesses = []
        else:
            if settings.ENVIRONMENT == "development":
                businesses = [
                    Business(
                        id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                        name="Development Retail Tenant",
                        slug="dev-retail",
                        plan_tier="starter",
                        retention_days=30,
                    )
                ]
            else:
                businesses = []

        for biz in businesses:
            cid = f"job-ml-forecast-{uuid.uuid4().hex[:8]}"
            set_correlation_id(cid)
            set_business_id(str(biz.id))

            start_time = time.perf_counter()
            logger.info("Starting ML %s job for tenant %s", job_name, biz.id)

            forecast_generated = False
            try:
                fc = await generate_forecast_for_business(biz.id)
                forecast_generated = fc is not None
            except Exception as e:
                logger.warning(
                    "Model execution failed for %s on tenant %s: %s",
                    job_name,
                    biz.id,
                    e,
                )
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "Completed ML %s job for tenant %s in %.2fms (generated=%s)",
                    job_name,
                    biz.id,
                    duration_ms,
                    forecast_generated,
                )
    finally:
        await redis_service.release_lock(lock_name)


async def run_retention_enforcement_job() -> None:
    """Scheduled job purging expired telemetry according to each tenant's plan retention policy."""
    job_name = "retention_enforcement"
    lock_name = f"ml_job:{job_name}"

    acquired = await redis_service.acquire_lock(lock_name, ttl_seconds=3500)
    if not acquired:
        logger.debug("Skipping %s job run: distributed lock already held by another worker", job_name)
        return

    try:
        if await is_db_available():
            try:
                async with AsyncSessionLocal() as session:
                    businesses = (await session.execute(select(Business))).scalars().all()
            except Exception as db_err:
                logger.warning("Could not query businesses for %s job: %s", job_name, db_err)
                businesses = []
        else:
            if settings.ENVIRONMENT == "development":
                businesses = [
                    Business(
                        id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                        name="Development Retail Tenant",
                        slug="dev-retail",
                        plan_tier="starter",
                        retention_days=30,
                    )
                ]
            else:
                businesses = []

        for biz in businesses:
            cid = f"job-ml-retention-{uuid.uuid4().hex[:8]}"
            set_correlation_id(cid)
            set_business_id(str(biz.id))

            start_time = time.perf_counter()
            logger.info("Starting %s job for tenant %s (retention: %d days)", job_name, biz.id, biz.retention_days or 30)

            deleted_count = 0
            try:
                deleted_count = await enforce_retention_for_business(biz)
            except Exception as e:
                logger.warning(
                    "Retention enforcement failed for tenant %s: %s",
                    biz.id,
                    e,
                )
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "Completed %s job for tenant %s in %.2fms (purged %d events)",
                    job_name,
                    biz.id,
                    duration_ms,
                    deleted_count,
                )
    finally:
        await redis_service.release_lock(lock_name)


class MLWorker:
    """Standalone background scheduler worker for ML jobs."""

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler()
        self.is_running = True

    def stop(self) -> None:
        logger.info("Received termination signal, shutting down ML jobs worker gracefully...")
        self.is_running = False
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def _register_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self.stop)
        except (NotImplementedError, AttributeError, ValueError):
            logger.info("Signal handlers not supported on this platform, using KeyboardInterrupt fallback")

    async def run(self) -> None:
        setup_logging()
        logger.info("Starting AI-CTO Background ML Jobs Worker...")

        self._register_signal_handlers()
        await redis_service.connect()

        # Schedule jobs
        # 1. Anomaly detection every 60 seconds
        self.scheduler.add_job(
            run_anomaly_detection_job,
            "interval",
            seconds=60,
            id="ml_anomaly_detection",
            replace_existing=True,
        )
        # 2. Forecasting every 5 minutes
        self.scheduler.add_job(
            run_forecasting_job,
            "interval",
            minutes=5,
            id="ml_forecasting",
            replace_existing=True,
        )
        # 3. Retention cleanup every 1 hour
        self.scheduler.add_job(
            run_retention_enforcement_job,
            "interval",
            hours=1,
            id="ml_retention_enforcement",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info("ML jobs scheduled successfully (anomaly: 60s, forecasting: 5m, retention: 1h)")

        while self.is_running:
            await asyncio.sleep(1.0)

        logger.info("Cleaning up resources on ML jobs worker exit...")
        await redis_service.disconnect()
        await engine.dispose()
        logger.info("ML jobs worker shutdown complete.")


def main():
    worker = MLWorker()
    try:
        asyncio.run(worker.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("ML worker interrupted, exiting.")


if __name__ == "__main__":
    main()
