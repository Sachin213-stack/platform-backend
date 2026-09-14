#!/usr/bin/env python3
"""
AICTO Synthetic Telemetry Data Generator
========================================
Memory-safe, streaming synthetic telemetry generator for AICTO.
Designed specifically for memory-constrained environments (e.g. 8GB RAM machines):
- Streams generation one (business, day) chunk at a time
- Sequential tenant processing (zero parallel contention)
- Uses np.float32 for numeric arrays (50% memory footprint)
- Explicit garbage collection & process RSS memory telemetry via ctypes/resource
- Reuses a single async session/connection for database insertion
- Generates physically-coupled metrics (load -> CPU -> queue -> latency)
- Injects labeled anomaly episodes and exports ground-truth JSON for model training

Usage:
    python -m scripts.generate_synthetic_telemetry [OPTIONS]

Example:
    python -m scripts.generate_synthetic_telemetry --days 7 --events-per-sec 0.35 --seed 42
"""

import argparse
import asyncio
import dataclasses
import gc
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sqlalchemy import select, delete, func, text
from sqlalchemy.dialects.postgresql import insert

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import setup_logging, logger
from app.db.session import AsyncSessionLocal, is_db_available, engine
from app.db.models.business import Business
from app.db.models.telemetry import TelemetryEvent


# ---------------------------------------------------------------------------
# Cross-Platform Process RSS Memory Tracker (Zero External Dependencies)
# ---------------------------------------------------------------------------

def get_process_memory_mb() -> float:
    """
    Returns current process Resident Set Size (RSS) in Megabytes.
    Uses standard library `resource` on Linux/macOS, and `ctypes` on Windows.
    Requires no third-party dependencies (no psutil).
    """
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    except ImportError:
        pass

    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        handle = k32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize / (1024.0 * 1024.0))
    except Exception:
        pass

    return 0.0


# ---------------------------------------------------------------------------
# Data Models for Anomaly Episodes & Generation Profiles
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AnomalyEpisode:
    episode_id: str
    business_id: str
    business_name: str
    anomaly_type: str
    metric: str
    severity: str
    start_time: str
    end_time: str
    start_dt: datetime
    end_dt: datetime
    duration_minutes: int
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "business_id": self.business_id,
            "business_name": self.business_name,
            "type": self.anomaly_type,
            "metric": self.metric,
            "severity": self.severity,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": self.duration_minutes,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclasses.dataclass
class BusinessRecord:
    id: uuid.UUID
    name: str
    slug: str
    plan_tier: str


@dataclasses.dataclass
class BusinessProfile:
    business_id: uuid.UUID
    name: str
    slug: str
    plan_tier: str
    business_type: str
    peak_hour: int
    weekend_factor: float
    base_latency: float
    base_cpu: float
    base_memory: float
    traffic_weight: float


ENDPOINTS_BY_TYPE = {
    "ecommerce": [
        ("/api/v1/products", 0.35, "GET"),
        ("/api/v1/search", 0.25, "GET"),
        ("/api/v1/cart", 0.18, "POST"),
        ("/api/v1/checkout", 0.12, "POST"),
        ("/api/v1/auth/login", 0.07, "POST"),
        ("/health", 0.03, "GET"),
    ],
    "saas": [
        ("/api/v1/dashboard/metrics", 0.35, "GET"),
        ("/api/v1/projects", 0.25, "GET"),
        ("/api/v1/tasks/execute", 0.20, "POST"),
        ("/api/v1/users/me", 0.12, "GET"),
        ("/api/v1/auth/token", 0.05, "POST"),
        ("/health", 0.03, "GET"),
    ],
    "media": [
        ("/api/v1/streams/manifest", 0.40, "GET"),
        ("/api/v1/catalog/featured", 0.28, "GET"),
        ("/api/v1/streams/heartbeat", 0.20, "POST"),
        ("/api/v1/users/profile", 0.07, "GET"),
        ("/api/v1/billing/subscribe", 0.03, "POST"),
        ("/health", 0.02, "GET"),
    ],
    "fintech": [
        ("/api/v1/transactions/list", 0.32, "GET"),
        ("/api/v1/accounts/balance", 0.28, "GET"),
        ("/api/v1/transfers/execute", 0.18, "POST"),
        ("/api/v1/cards/authorize", 0.12, "POST"),
        ("/api/v1/auth/mfa/verify", 0.07, "POST"),
        ("/health", 0.03, "GET"),
    ],
}


def create_tenant_profile(biz: BusinessRecord, idx: int) -> BusinessProfile:
    """Assigns realistic behavioral parameters per business."""
    slug = biz.slug.lower()
    if "retail" in slug or "apex" in slug:
        btype = "ecommerce"
        peak = 20  # Evening shopping surge
        wf = 1.35  # Weekends surge
    elif "cloud" in slug or "crm" in slug or "nexus" in slug:
        btype = "saas"
        peak = 14  # Business hours
        wf = 0.40  # Weekends drop sharply
    elif "media" in slug or "stream" in slug or "games" in slug or "vortex" in slug:
        btype = "media"
        peak = 21  # Prime-time entertainment peak
        wf = 1.50  # Heavy weekend streaming
    elif "fin" in slug or "pay" in slug or "cyber" in slug:
        btype = "fintech"
        peak = 13  # Midday banking activity
        wf = 0.65  # Modest weekend volume
    else:
        types = ["ecommerce", "saas", "media", "fintech"]
        btype = types[idx % len(types)]
        peak = (12 + idx * 2) % 24
        wf = 0.8 + (idx % 4) * 0.2

    tier = (biz.plan_tier or "starter").lower()
    weight = 1.6 if "enterprise" in tier else 1.0 if "growth" in tier else 0.55

    return BusinessProfile(
        business_id=biz.id,
        name=biz.name,
        slug=biz.slug,
        plan_tier=tier,
        business_type=btype,
        peak_hour=peak,
        weekend_factor=wf,
        base_latency=45.0 + (idx % 5) * 15.0,
        base_cpu=22.0 + (idx % 4) * 4.0,
        base_memory=42.0 + (idx % 3) * 6.0,
        traffic_weight=weight,
    )


# ---------------------------------------------------------------------------
# Anomaly Schedule Generation (~3% to 8% Contamination)
# ---------------------------------------------------------------------------

def generate_anomaly_schedule_for_business(
    profile: BusinessProfile,
    start_time: datetime,
    end_time: datetime,
    rng: np.random.Generator,
    contamination: float = 0.05,
) -> List[AnomalyEpisode]:
    """
    Plans non-overlapping anomaly episodes across the timeline for a given tenant,
    targeting approximately `contamination` fraction of total duration.
    """
    total_minutes = int((end_time - start_time).total_seconds() / 60)
    target_anomaly_minutes = int(total_minutes * contamination)

    episodes: List[AnomalyEpisode] = []
    used_windows: List[Tuple[datetime, datetime]] = []

    anomaly_types = [
        ("latency_spike", "response_time", [15, 20, 25, 30]),
        ("cpu_saturation", "cpu_usage", [20, 30, 45]),
        ("queue_explosion", "queue_depth", [15, 20, 30]),
        ("memory_leak", "memory_usage", [90, 120, 180, 240]),
    ]

    accumulated_minutes = 0
    max_attempts = 150
    attempts = 0

    while accumulated_minutes < target_anomaly_minutes and attempts < max_attempts:
        attempts += 1
        atype, metric, duration_choices = anomaly_types[rng.integers(0, len(anomaly_types))]
        dur = int(rng.choice(duration_choices))

        offset_min = int(rng.integers(60, max(61, total_minutes - dur - 60)))
        ep_start = start_time + timedelta(minutes=offset_min)
        ep_end = ep_start + timedelta(minutes=dur)

        # Check collision with existing episodes (30m gap)
        collides = False
        for s, e in used_windows:
            if not (ep_end < s - timedelta(minutes=30) or ep_start > e + timedelta(minutes=30)):
                collides = True
                break

        if collides:
            continue

        used_windows.append((ep_start, ep_end))
        accumulated_minutes += dur

        if atype == "latency_spike":
            mult = round(float(rng.uniform(3.5, 7.5)), 2)
            sev = "critical" if mult >= 5.0 else "high"
            desc = f"Database connection pool saturation causing {mult}x response latency spike on checkout/search endpoints"
            params = {"multiplier": mult, "target_endpoints": ["/api/v1/checkout", "/api/v1/search"]}
        elif atype == "cpu_saturation":
            sustained_cpu = round(float(rng.uniform(94.5, 99.2)), 1)
            sev = "critical"
            desc = f"Worker thread deadlock resulting in sustained {sustained_cpu}% CPU saturation and thread queuing"
            params = {"sustained_cpu_pct": sustained_cpu}
        elif atype == "queue_explosion":
            peak_q = int(rng.integers(70, 210))
            sev = "critical" if peak_q >= 120 else "high"
            desc = f"Upstream service degradation: ingress queue backlogged to {peak_q} items with elevated 504 gateway timeouts"
            params = {"peak_queue_depth": peak_q}
        else:  # memory_leak
            leak_peak = round(float(rng.uniform(94.0, 98.8)), 1)
            sev = "high"
            desc = f"Unbounded in-memory session cache leak gradually consuming {leak_peak}% RAM over {dur} minutes"
            params = {"peak_memory_pct": leak_peak}

        episodes.append(
            AnomalyEpisode(
                episode_id=str(uuid.uuid4()),
                business_id=str(profile.business_id),
                business_name=profile.name,
                anomaly_type=atype,
                metric=metric,
                severity=sev,
                start_time=ep_start.isoformat(),
                end_time=ep_end.isoformat(),
                start_dt=ep_start,
                end_dt=ep_end,
                duration_minutes=dur,
                description=desc,
                parameters=params,
            )
        )

    episodes.sort(key=lambda x: x.start_dt)
    return episodes


# ---------------------------------------------------------------------------
# Physics & Correlated Metric Engine (using np.float32)
# ---------------------------------------------------------------------------

def calculate_load_factor(dt: datetime, profile: BusinessProfile) -> float:
    """Computes realistic 0.0 - 1.0 traffic load factor."""
    hour_float = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    phase = (hour_float - profile.peak_hour) * (2 * np.pi / 24.0)
    diurnal = 0.52 + 0.40 * np.cos(phase)

    # Midday secondary wave
    midday_phase = (hour_float - 13.0) * (4 * np.pi / 24.0)
    diurnal += 0.06 * np.cos(midday_phase)

    is_weekend = dt.weekday() >= 5
    day_multiplier = profile.weekend_factor if is_weekend else 1.0

    load = diurnal * day_multiplier
    return float(np.clip(load, 0.08, 1.05))


def generate_single_day_chunk(
    profile: BusinessProfile,
    day_start: datetime,
    day_end: datetime,
    n_events: int,
    episodes: List[AnomalyEpisode],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Generates a single (business, day) chunk of telemetry events in memory.
    Uses np.float32 for numerical arrays to minimize memory footprint.
    """
    endpoints_config = ENDPOINTS_BY_TYPE.get(profile.business_type, ENDPOINTS_BY_TYPE["ecommerce"])
    endpoint_urls = [ep[0] for ep in endpoints_config]
    endpoint_probs = [ep[1] for ep in endpoints_config]

    # Hourly density for this single 24-hour day
    hour_timestamps = [day_start + timedelta(hours=h) for h in range(24)]
    hourly_densities = np.array([calculate_load_factor(dt, profile) for dt in hour_timestamps], dtype=np.float32)
    hourly_probs = hourly_densities / np.sum(hourly_densities)

    # Sample timestamps within the single day
    sampled_hours = rng.choice(24, size=n_events, p=hourly_probs)
    second_offsets = rng.uniform(0, 3600, size=n_events).astype(np.float32)

    timestamps = [
        hour_timestamps[sampled_hours[k]] + timedelta(seconds=float(second_offsets[k]))
        for k in range(n_events)
    ]
    timestamps.sort()

    chosen_endpoints = rng.choice(endpoint_urls, size=n_events, p=endpoint_probs)

    # Find active episodes overlapping this day
    day_episodes = [ep for ep in episodes if not (ep.end_dt < day_start or ep.start_dt > day_end)]

    records: List[Dict[str, Any]] = []
    ep_idx = 0
    num_day_episodes = len(day_episodes)

    for i, ts in enumerate(timestamps):
        while ep_idx < num_day_episodes and day_episodes[ep_idx].end_dt < ts:
            ep_idx += 1

        active_ep: Optional[AnomalyEpisode] = None
        if ep_idx < num_day_episodes and day_episodes[ep_idx].start_dt <= ts <= day_episodes[ep_idx].end_dt:
            active_ep = day_episodes[ep_idx]

        load = float(calculate_load_factor(ts, profile))
        load_jitter = float(rng.normal(0.0, 0.04))
        effective_load = float(np.clip(load + load_jitter, 0.05, 1.15))

        endpoint = str(chosen_endpoints[i])
        is_checkout = "/checkout" in endpoint
        is_cart = "/cart" in endpoint

        # Float32 physical coupling math
        cpu_val = float(profile.base_cpu + (effective_load ** 1.3) * 45.0 + rng.normal(0, 2.5))
        cpu_val = float(np.clip(cpu_val, 8.0, 92.0))

        mem_val = float(profile.base_memory + effective_load * 12.0 + rng.normal(0, 1.2))
        mem_val = float(np.clip(mem_val, 25.0, 88.0))

        if cpu_val > 72.0:
            queue_val = int(rng.poisson(lam=max(1.0, (cpu_val - 70.0) * 0.45)))
        else:
            queue_val = int(rng.choice([0, 1, 2], p=[0.75, 0.20, 0.05]))

        cpu_penalty = (cpu_val / 100.0) ** 2.2 * 85.0
        queue_penalty = queue_val * 4.5
        latency_val = float(profile.base_latency + cpu_penalty + queue_penalty + rng.lognormal(mean=1.5, sigma=0.35))

        orders_count = 0
        revenue_val = 0.0
        if is_checkout:
            orders_count = int(rng.poisson(lam=max(0.4, effective_load * 2.2)))
            if orders_count > 0:
                order_amounts = rng.lognormal(mean=4.2, sigma=0.6, size=orders_count).astype(np.float32)
                revenue_val = round(float(np.sum(order_amounts)), 2)
        elif is_cart and rng.random() < 0.15:
            orders_count = 1
            revenue_val = round(float(rng.lognormal(mean=3.8, sigma=0.5)), 2)

        if queue_val > 40 or cpu_val > 90.0:
            status_code = int(rng.choice([200, 500, 502, 504], p=[0.65, 0.15, 0.08, 0.12]))
        else:
            status_code = int(rng.choice([200, 400, 404, 429, 500], p=[0.992, 0.003, 0.003, 0.001, 0.001]))

        # Anomaly overrides
        if active_ep is not None:
            if active_ep.anomaly_type == "latency_spike":
                mult = float(active_ep.parameters.get("multiplier", 4.5))
                if endpoint in active_ep.parameters.get("target_endpoints", [endpoint]):
                    latency_val = latency_val * mult + float(rng.uniform(80.0, 250.0))
                else:
                    latency_val = latency_val * (1.0 + (mult - 1.0) * 0.4)

            elif active_ep.anomaly_type == "cpu_saturation":
                target_cpu = float(active_ep.parameters.get("sustained_cpu_pct", 96.5))
                cpu_val = float(np.clip(target_cpu + float(rng.normal(0, 1.0)), 93.0, 99.8))
                queue_val = max(int(queue_val), int(rng.integers(35, 95)))
                latency_val = latency_val * float(rng.uniform(2.2, 3.8))
                if rng.random() < 0.25:
                    status_code = int(rng.choice([500, 503, 504]))

            elif active_ep.anomaly_type == "queue_explosion":
                peak_q = int(active_ep.parameters.get("peak_queue_depth", 120))
                queue_val = max(45, int(rng.normal(peak_q, 15)))
                latency_val += queue_val * 6.5
                if rng.random() < 0.35:
                    status_code = 504

            elif active_ep.anomaly_type == "memory_leak":
                ep_progress = (ts - active_ep.start_dt).total_seconds() / (active_ep.end_dt - active_ep.start_dt).total_seconds()
                ep_progress = float(np.clip(ep_progress, 0.0, 1.0))
                peak_mem = float(active_ep.parameters.get("peak_memory_pct", 96.0))
                mem_val = profile.base_memory + ep_progress * (peak_mem - profile.base_memory) + float(rng.normal(0, 0.5))
                mem_val = float(np.clip(mem_val, 40.0, 99.5))

        event_type = "order" if orders_count > 0 else "error" if status_code >= 500 else "request"

        records.append({
            "id": uuid.uuid4(),
            "business_id": profile.business_id,
            "idempotency_key": str(uuid.uuid4()),
            "event_type": str(event_type),
            "response_time_ms": round(float(latency_val), 1),
            "status_code": int(status_code),
            "orders_count": int(orders_count),
            "revenue_amount": round(float(revenue_val), 2),
            "cpu_usage_pct": round(float(cpu_val), 1),
            "memory_usage_pct": round(float(mem_val), 1),
            "queue_depth": int(queue_val),
            "endpoint": str(endpoint),
            "payload_metadata": {
                "env": "production",
                "region": "us-east-1",
                "client": "web-sdk",
                "anomaly_active": bool(active_ep is not None),
            },
            "timestamp": ts,
        })

    return records


# ---------------------------------------------------------------------------
# Streamed Execution Loop (One Business, One Day at a Time)
# ---------------------------------------------------------------------------

async def run_generator(
    days: int,
    events_per_sec: float,
    business_filter: str,
    seed: int,
    output_dir: str,
    batch_size: int,
    wipe: bool,
) -> None:
    start_wall_time = time.perf_counter()
    setup_logging()
    logger.info("=" * 78)
    logger.info("AICTO MEMORY-SAFE SYNTHETIC TELEMETRY GENERATOR")
    logger.info("=" * 78)

    initial_rss = get_process_memory_mb()
    logger.info("Hardware Profile: 8GB RAM Safe Mode (Streaming 1-tenant, 1-day chunks)")
    logger.info("Initial Process RSS Memory: %.2f MB", initial_rss)

    if not await is_db_available(force_check=True):
        logger.error("Database connection failed. Ensure local Docker PostgreSQL is running on port 5432.")
        sys.exit(1)

    rng = np.random.default_rng(seed)

    # 1. Fetch businesses (Only existing columns: id, name, slug, plan_tier)
    async with AsyncSessionLocal() as session:
        stmt = select(Business.id, Business.name, Business.slug, Business.plan_tier)
        if business_filter and business_filter.lower() != "all":
            filter_tokens = [t.strip() for t in business_filter.split(",")]
            biz_conditions = []
            for token in filter_tokens:
                try:
                    biz_conditions.append(Business.id == uuid.UUID(token))
                except ValueError:
                    biz_conditions.append(Business.slug == token)
            from sqlalchemy import or_
            stmt = stmt.where(or_(*biz_conditions))

        result = await session.execute(stmt)
        raw_rows = result.all()
        businesses = [
            BusinessRecord(id=r[0], name=r[1], slug=r[2], plan_tier=r[3])
            for r in raw_rows
        ]

    if not businesses:
        logger.error("No businesses found matching filter '%s'.", business_filter)
        sys.exit(1)

    logger.info("Identified %d business tenant(s) for generation.", len(businesses))
    profiles = [create_tenant_profile(b, idx) for idx, b in enumerate(businesses)]

    # 2. Time Horizon & Targets
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    timeline_start = now_utc - timedelta(days=days)
    timeline_seconds = days * 86400

    target_total_events = int(timeline_seconds * events_per_sec)
    logger.info(
        "Timeline: %s -> %s (%d days) | Target Rate: %.2f ev/s | Target Volume: %s events",
        timeline_start.isoformat(),
        now_utc.isoformat(),
        days,
        events_per_sec,
        f"{target_total_events:,}",
    )

    # 3. Handle --wipe if requested
    if wipe:
        target_ids = [p.business_id for p in profiles]
        logger.warning("WIPE REQUESTED: Purging existing telemetry_events for %d businesses...", len(target_ids))
        async with AsyncSessionLocal() as session:
            del_stmt = delete(TelemetryEvent).where(TelemetryEvent.business_id.in_(target_ids))
            del_res = await session.execute(del_stmt)
            await session.commit()
            logger.warning("Wipe complete: deleted %d existing telemetry rows.", del_res.rowcount or 0)

    # 4. Generate Anomaly Schedules & Export Ground Truth
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ground_truth_file = out_path / "anomaly_ground_truth.json"

    all_episodes: List[AnomalyEpisode] = []
    episodes_by_biz: Dict[str, List[AnomalyEpisode]] = {}

    for profile in profiles:
        eps = generate_anomaly_schedule_for_business(profile, timeline_start, now_utc, rng, contamination=0.05)
        episodes_by_biz[str(profile.business_id)] = eps
        all_episodes.extend(eps)

    # Write ground truth JSON file
    gt_data = [ep.to_dict() for ep in all_episodes]
    with open(ground_truth_file, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)

    logger.info(
        "Exported ground truth: %d total anomaly episodes -> %s",
        len(all_episodes),
        ground_truth_file,
    )

    # 5. Distribute Target Event Counts per Business
    total_weight = sum(p.traffic_weight for p in profiles)
    biz_daily_counts: Dict[str, int] = {}
    total_planned = 0

    for p in profiles:
        biz_total = int(round(target_total_events * (p.traffic_weight / total_weight)))
        daily_count = max(100, int(round(biz_total / days)))
        biz_daily_counts[str(p.business_id)] = daily_count
        total_planned += daily_count * days

    logger.info(
        "Streaming Plan: %d tenants x %d days = %d chunks (avg ~%d events/chunk) -> Total planned: %s",
        len(profiles),
        days,
        len(profiles) * days,
        int(total_planned / max(1, len(profiles) * days)),
        f"{total_planned:,}",
    )

    # 6. Stream Generation & Batch Inserts (Sequential single connection)
    total_inserted = 0
    peak_rss_mb = initial_rss
    db_write_time = 0.0
    total_chunks = len(profiles) * days
    chunk_counter = 0

    print("\nStreaming generation and PostgreSQL insertion (one chunk at a time)...")
    sys.stdout.flush()

    # Reuse a single async session/connection for the entire streaming run
    async with AsyncSessionLocal() as session:
        for p_idx, profile in enumerate(profiles):
            biz_id_str = str(profile.business_id)
            daily_n = biz_daily_counts[biz_id_str]
            eps = episodes_by_biz[biz_id_str]

            for day_idx in range(days):
                chunk_counter += 1
                day_start = timeline_start + timedelta(days=day_idx)
                day_end = day_start + timedelta(days=1)

                # Generate ONLY this single (business, day) in RAM
                chunk_records = generate_single_day_chunk(
                    profile=profile,
                    day_start=day_start,
                    day_end=day_end,
                    n_events=daily_n,
                    episodes=eps,
                    rng=rng,
                )

                # Insert in slices of batch_size
                for b_start in range(0, len(chunk_records), batch_size):
                    slice_records = chunk_records[b_start : b_start + batch_size]
                    t_ins = time.perf_counter()
                    stmt = insert(TelemetryEvent).values(slice_records)
                    await session.execute(stmt)
                    await session.commit()
                    db_write_time += (time.perf_counter() - t_ins)
                    total_inserted += len(slice_records)

                # Explicitly free memory for this chunk
                del chunk_records
                gc.collect()

                # Memory check & Progress update
                current_rss = get_process_memory_mb()
                if current_rss > peak_rss_mb:
                    peak_rss_mb = current_rss

                if current_rss > 1500.0:
                    logger.warning("HIGH MEMORY WARNING: Process RSS exceeded 1.5GB (Current: %.2f MB)", current_rss)

                pct = (chunk_counter / total_chunks) * 100.0
                elapsed = time.perf_counter() - start_wall_time
                rate = total_inserted / max(0.1, elapsed)

                print(
                    f"\r  Chunk [{chunk_counter:>3}/{total_chunks}] ({pct:5.1f}%) | "
                    f"Tenant: {profile.slug:<16} Day {day_idx + 1}/{days} | "
                    f"Events: {total_inserted:>8,} | "
                    f"RSS: {current_rss:5.1f} MB (Peak: {peak_rss_mb:5.1f} MB) | "
                    f"Rate: {rate:6,.0f} ev/s",
                    end="",
                    flush=True,
                )

    total_wall_time = time.perf_counter() - start_wall_time
    print("\n" + "=" * 78)
    logger.info("GENERATION AND INSERTION COMPLETE!")
    logger.info(
        "Inserted %s events in %.2fs (DB Write: %.2fs, Net Throughput: %.0f ev/s)",
        f"{total_inserted:,}",
        total_wall_time,
        db_write_time,
        total_inserted / max(0.1, total_wall_time),
    )
    logger.info("Memory Profile: Initial RSS = %.2f MB | Peak RSS = %.2f MB", initial_rss, peak_rss_mb)
    logger.info("=" * 78)

    # 7. Verification Queries against Database
    logger.info("Running SQL verification checks on PostgreSQL...")
    async with AsyncSessionLocal() as session:
        agg_stmt = select(
            func.count(TelemetryEvent.id).label("total_rows"),
            func.min(TelemetryEvent.timestamp).label("min_ts"),
            func.max(TelemetryEvent.timestamp).label("max_ts"),
            func.avg(TelemetryEvent.response_time_ms).label("avg_rt"),
            func.avg(TelemetryEvent.cpu_usage_pct).label("avg_cpu"),
            func.avg(TelemetryEvent.memory_usage_pct).label("avg_mem"),
            func.sum(TelemetryEvent.orders_count).label("total_orders"),
            func.sum(TelemetryEvent.revenue_amount).label("total_rev"),
        )
        agg_res = (await session.execute(agg_stmt)).first()

        biz_stmt = (
            select(
                TelemetryEvent.business_id,
                func.count(TelemetryEvent.id).label("cnt"),
                func.avg(TelemetryEvent.response_time_ms).label("avg_rt"),
                func.avg(TelemetryEvent.cpu_usage_pct).label("avg_cpu"),
            )
            .group_by(TelemetryEvent.business_id)
            .order_by(func.count(TelemetryEvent.id).desc())
        )
        biz_res = (await session.execute(biz_stmt)).all()

    print("\n--- SQL VERIFICATION REPORT ---")
    print(f"Total Rows in DB : {agg_res.total_rows:,}")
    print(f"Earliest Event   : {agg_res.min_ts}")
    print(f"Latest Event     : {agg_res.max_ts}")
    print(f"Average Latency  : {agg_res.avg_rt:.2f} ms")
    print(f"Average CPU      : {agg_res.avg_cpu:.2f} %")
    print(f"Average Memory   : {agg_res.avg_mem:.2f} %")
    print(f"Total Orders     : {int(agg_res.total_orders or 0):,}")
    print(f"Total Revenue    : ${float(agg_res.total_rev or 0.0):,.2f}")
    print(f"Peak Process RSS : {peak_rss_mb:.2f} MB")
    print("\nPer-Business Row Distribution:")
    for row in biz_res:
        name_match = next((p.name for p in profiles if p.business_id == row.business_id), "Unknown")
        print(f"  {str(row.business_id):<36} | {name_match:<24} | {row.cnt:>8,} events | RT: {row.avg_rt:5.1f}ms | CPU: {row.avg_cpu:4.1f}%")

    print("\nGround Truth Anomaly Episodes Summary:")
    by_type: Dict[str, int] = {}
    by_sev: Dict[str, int] = {}
    for ep in all_episodes:
        by_type[ep.anomaly_type] = by_type.get(ep.anomaly_type, 0) + 1
        by_sev[ep.severity] = by_sev.get(ep.severity, 0) + 1

    print(f"  Total Injected Episodes : {len(all_episodes)}")
    print("  By Anomaly Type         :", by_type)
    print("  By Severity             :", by_sev)
    print(f"  Ground Truth File       : {ground_truth_file.resolve()}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="AICTO Memory-Safe Synthetic Telemetry Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hardware-safe reduced defaults
    parser.add_argument("--days", type=int, default=7, help="Number of historical days to generate (default: 7)")
    parser.add_argument("--events-per-sec", type=float, default=0.35, help="Target average event arrival rate across all businesses (targets ~210K events)")
    parser.add_argument("--businesses", type=str, default="all", help="Target businesses: 'all' or comma-separated UUIDs/slugs")
    parser.add_argument("--seed", type=int, default=42, help="NumPy random generator seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="scripts/output", help="Directory to save ground truth JSON and verification artifacts")
    parser.add_argument("--batch-size", type=int, default=1000, help="Transaction batch size for PostgreSQL asyncpg inserts (default: 1000)")
    parser.add_argument("--wipe", action="store_true", help="Purge existing telemetry_events for targeted businesses before generation")

    args = parser.parse_args()

    async def _runner():
        try:
            await run_generator(
                days=args.days,
                events_per_sec=args.events_per_sec,
                business_filter=args.businesses,
                seed=args.seed,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                wipe=args.wipe,
            )
        finally:
            await engine.dispose()

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        print("\nGenerator interrupted by user.")


if __name__ == "__main__":
    main()
