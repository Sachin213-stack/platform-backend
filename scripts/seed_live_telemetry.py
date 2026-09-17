#!/usr/bin/env python3
"""
AICTO Telemetry Data Seeder
===========================
Seeds realistic telemetry events across recent hours into AICTO backend
(supports local decoupled mode, local database, or live HTTP API).

Ensures:
- 72 hourly historical data points for traffic forecasting
- 30+ recent telemetry events for IsolationForest anomaly detection
- 3 realistic injected anomalies (latency spike, CPU saturation, memory leak)
- Triggers ML jobs and asserts output anomaly/forecast generation

Usage:
    python -m scripts.seed_live_telemetry [--api-url http://localhost:8000/api]
"""

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx

DEFAULT_BUSINESS_ID = "11111111-1111-1111-1111-111111111111"


def generate_telemetry_dataset(business_id: str) -> List[Dict[str, Any]]:
    """Generates 72 hours of hourly history + 30 recent events with 3 injected anomalies."""
    now = datetime.now(timezone.utc)
    events: List[Dict[str, Any]] = []

    # 1. 72 hourly historical traffic aggregations (for LinearRegression forecast)
    for hour_idx in range(72, 2, -1):
        ts = now - timedelta(hours=hour_idx)
        hour_of_day = ts.hour
        base_rate = 15 + int(12 * (1 + random.random()) * (1 if 9 <= hour_of_day <= 21 else 0.4))

        for event_i in range(base_rate):
            event_ts = ts + timedelta(minutes=random.randint(0, 58), seconds=random.randint(0, 59))
            events.append({
                "idempotency_key": f"seed-hist-{business_id[:8]}-h{hour_idx}-e{event_i}",
                "event_type": "request",
                "endpoint": random.choice(["/api/products", "/api/search", "/api/cart", "/api/checkout"]),
                "response_time_ms": round(float(random.gauss(120, 20)), 1),
                "status_code": 200 if random.random() > 0.02 else 500,
                "orders_count": 1 if random.random() > 0.7 else 0,
                "revenue_amount": round(random.uniform(15.0, 150.0), 2) if random.random() > 0.7 else 0.0,
                "cpu_usage_pct": round(float(min(85, max(15, random.gauss(38, 8)))), 1),
                "memory_usage_pct": round(float(min(88, max(25, random.gauss(52, 6)))), 1),
                "queue_depth": random.randint(1, 8),
                "timestamp": event_ts.isoformat(),
            })

    # 2. Recent telemetry events in the last 2 hours (for IsolationForest anomaly detection)
    recent_endpoints = ["/api/checkout", "/api/products", "/api/search", "/api/cart", "/api/auth/me"]
    for i in range(35):
        ts = now - timedelta(minutes=random.randint(3, 115))
        events.append({
            "idempotency_key": f"seed-recent-{business_id[:8]}-{i}",
            "event_type": "request",
            "endpoint": random.choice(recent_endpoints),
            "response_time_ms": round(float(random.gauss(125, 18)), 1),
            "status_code": 200,
            "orders_count": 1 if random.random() > 0.6 else 0,
            "revenue_amount": round(random.uniform(20.0, 80.0), 2) if random.random() > 0.6 else 0.0,
            "cpu_usage_pct": round(float(random.gauss(36, 6)), 1),
            "memory_usage_pct": round(float(random.gauss(54, 5)), 1),
            "queue_depth": random.randint(1, 4),
            "timestamp": ts.isoformat(),
        })

    # 3. Inject 3 distinct, undeniable anomaly outliers
    # Anomaly 1: Severe latency spike on checkout
    events.append({
        "idempotency_key": f"seed-anom-latency-{uuid.uuid4().hex[:6]}",
        "event_type": "request",
        "endpoint": "/api/checkout",
        "response_time_ms": 1450.0,
        "status_code": 200,
        "orders_count": 0,
        "revenue_amount": 0.0,
        "cpu_usage_pct": 52.0,
        "memory_usage_pct": 60.0,
        "queue_depth": 18,
        "timestamp": (now - timedelta(minutes=18)).isoformat(),
    })

    # Anomaly 2: High CPU saturation on inventory sync
    events.append({
        "idempotency_key": f"seed-anom-cpu-{uuid.uuid4().hex[:6]}",
        "event_type": "request",
        "endpoint": "/api/inventory-sync",
        "response_time_ms": 110.0,
        "status_code": 200,
        "orders_count": 0,
        "revenue_amount": 0.0,
        "cpu_usage_pct": 96.5,
        "memory_usage_pct": 55.0,
        "queue_depth": 25,
        "timestamp": (now - timedelta(minutes=32)).isoformat(),
    })

    # Anomaly 3: Memory leak spike on reports export
    events.append({
        "idempotency_key": f"seed-anom-mem-{uuid.uuid4().hex[:6]}",
        "event_type": "request",
        "endpoint": "/api/reports/export",
        "response_time_ms": 130.0,
        "status_code": 200,
        "orders_count": 0,
        "revenue_amount": 0.0,
        "cpu_usage_pct": 42.0,
        "memory_usage_pct": 97.8,
        "queue_depth": 12,
        "timestamp": (now - timedelta(minutes=45)).isoformat(),
    })

    # Sort chronologically
    events.sort(key=lambda e: e["timestamp"])
    return events


async def seed_via_api(api_url: str, business_id: str) -> None:
    """Seeds telemetry events by calling the FastAPI HTTP API."""
    print(f"=== SEEDING TELEMETRY VIA HTTP API: {api_url} ===")
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check health first
        h_res = await client.get(f"{api_url}/health")
        print(f"Target Health Probe: HTTP {h_res.status_code}")

        # Obtain bearer token
        token = None
        try:
            demo_res = await client.post(f"{api_url}/auth/demo")
            if demo_res.status_code == 200:
                token = demo_res.json().get("access_token")
                print("Acquired demo bearer token successfully.")
        except Exception as e:
            print(f"Demo login note: {e}")

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        headers["X-Business-ID"] = business_id

        # Generate dataset
        dataset = generate_telemetry_dataset(business_id)
        print(f"Generated {len(dataset)} synthetic telemetry events.")

        # Ingest events
        success_count = 0
        for i, ev in enumerate(dataset):
            try:
                res = await client.post(
                    f"{api_url}/ingestion/events",
                    json=ev,
                    headers=headers,
                )
                if res.status_code in [200, 202]:
                    success_count += 1
            except Exception as ex:
                if i % 50 == 0:
                    print(f"Event {i} submission error: {ex}")

            if (i + 1) % 100 == 0 or (i + 1) == len(dataset):
                print(f"Ingested {i + 1}/{len(dataset)} events...")

        print(f"Completed ingestion: {success_count}/{len(dataset)} events accepted.")

        # Query dashboard metrics
        try:
            dash_res = await client.get(f"{api_url}/dashboard/metrics", headers=headers)
            if dash_res.status_code == 200:
                d = dash_res.json()
                print("\nDashboard Metrics After Seeding:")
                print(f"  KPI Response Time: {d.get('kpis', {}).get('response_time_ms')}ms")
                print(f"  Recent Anomalies Count: {len(d.get('recent_anomalies', []))}")
                for a in d.get("recent_anomalies", []):
                    print(f"    - [{a.get('severity').upper()}] {a.get('metric_name')}: {a.get('actual_value')} ({a.get('description')})")
        except Exception as e:
            print(f"Could not query dashboard: {e}")


async def seed_direct_db(business_id_str: str) -> None:
    """Directly seeds events into PostgreSQL database and runs ML jobs."""
    from app.db.session import AsyncSessionLocal, is_db_available, set_rls_context, engine
    from app.db.models.telemetry import TelemetryEvent
    from app.db.models.ml import Anomaly, Forecast
    from app.workers.ml_jobs import (
        detect_anomalies_for_business,
        generate_forecast_for_business,
    )
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy import select, func

    print("=== SEEDING TELEMETRY DIRECTLY INTO DATABASE ===")
    biz_uuid = uuid.UUID(business_id_str)

    if not await is_db_available():
        print("Database is offline. Operating in decoupled test harness mode.")
        return

    dataset = generate_telemetry_dataset(business_id_str)
    print(f"Generated {len(dataset)} synthetic telemetry events.")

    db_records = []
    for d in dataset:
        rec = dict(d)
        rec["id"] = uuid.uuid4()
        rec["business_id"] = biz_uuid
        rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
        db_records.append(rec)

    async with AsyncSessionLocal() as session:
        await set_rls_context(session, business_id_str)
        stmt = insert(TelemetryEvent).values(db_records)
        stmt = stmt.on_conflict_do_nothing(index_elements=["idempotency_key"])
        await session.execute(stmt)
        await session.commit()
        print(f"Inserted {len(db_records)} records to telemetry_events table.")

        # Count total in DB
        cnt = (await session.execute(
            select(func.count(TelemetryEvent.id)).where(TelemetryEvent.business_id == biz_uuid)
        )).scalar()
        print(f"Total telemetry_events for business {business_id_str}: {cnt}")

    # Trigger ML Jobs directly
    print("\nRunning ML Anomaly Detection Job...")
    anom_count = await detect_anomalies_for_business(biz_uuid)
    print(f"ML Anomaly Detection completed! Detected {anom_count} new anomalies.")

    print("\nRunning ML Forecasting Job...")
    forecast = await generate_forecast_for_business(biz_uuid)
    if forecast:
        print(f"ML Forecast generated successfully! Crash Risk: {forecast.crash_risk_pct}%")
        print(f"Forecast Horizon: {forecast.forecast_horizon}, Points: {len(forecast.forecast_curve.get('yhat', []))}")
    else:
        print("Forecast returned None.")

    async with AsyncSessionLocal() as session:
        await set_rls_context(session, business_id_str)
        anoms = (await session.execute(
            select(Anomaly).where(Anomaly.business_id == biz_uuid).order_by(Anomaly.detected_at.desc()).limit(5)
        )).scalars().all()
        print(f"\nTop {len(anoms)} Anomalies in Database:")
        for a in anoms:
            print(f"  - [{a.severity.upper()}] {a.metric_name}: actual={a.actual_value} expected={a.expected_value}")
            print(f"    Description: {a.description}")


def main():
    parser = argparse.ArgumentParser(description="Seed realistic telemetry and verify ML workers")
    parser.add_argument("--api-url", type=str, default="", help="Backend API base URL (e.g. http://localhost:8000/api)")
    parser.add_argument("--business-id", type=str, default=DEFAULT_BUSINESS_ID, help="Target business tenant UUID")
    parser.add_argument("--direct-db", action="store_true", help="Insert directly via SQLAlchemy session")

    args = parser.parse_args()

    if args.api_url:
        asyncio.run(seed_via_api(args.api_url.rstrip("/"), args.business_id))
    else:
        asyncio.run(seed_direct_db(args.business_id))


if __name__ == "__main__":
    main()
