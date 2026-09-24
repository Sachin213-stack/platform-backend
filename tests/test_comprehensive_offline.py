import sys
import uuid
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.security import create_access_token
from app.services.ingestion_service import IngestionService
from app.workers.ml_jobs import (
    run_anomaly_detection_job,
    run_forecasting_job,
    run_retention_enforcement_job,
)

def test_all():
    setup_logging()
    client = TestClient(app)
    print("=== TESTING ALL ENDPOINTS & JOBS IN DECOUPLED OFFLINE MODE ===")

    # 1. Root & Health
    r = client.get("/")
    assert r.status_code == 200, f"Root failed: {r.text}"
    print("[PASS] GET / -> 200")

    r = client.get("/api/health")
    assert r.status_code == 200, f"Health failed: {r.text}"
    data = r.json()
    assert data["status"] == "degraded", f"Expected degraded status, got {data['status']}"
    assert data["services"]["postgresql"]["status"] == "degraded"
    print(f"[PASS] GET /api/health -> 200 (status={data['status']}, pg={data['services']['postgresql']['status']})")

    # 2. Auth - Register & Login & Me & Logout
    reg_payload = {
        "email": "dev_test_user@example.com",
        "password": "SecurePassword123!",
        "full_name": "Dev User",
        "business_name": "Apex Test Retail",
    }
    r = client.post("/api/auth/register", json=reg_payload)
    assert r.status_code in [201, 200], f"Register failed: {r.text}"
    token = r.json().get("access_token")
    assert token, "No token returned from register"
    print("[PASS] POST /api/auth/register -> 201/200 (decoupled dev user created)")

    login_payload = {
        "email": "dev_test_user@example.com",
        "password": "SecurePassword123!",
    }
    r = client.post("/api/auth/login", json=login_payload)
    assert r.status_code == 200, f"Login failed: {r.text}"
    auth_header = {"Authorization": f"Bearer {token}"}
    print("[PASS] POST /api/auth/login -> 200")

    r = client.get("/api/auth/me", headers=auth_header)
    assert r.status_code == 200, f"/me failed: {r.text}"
    me_data = r.json()
    assert me_data["email"] == "dev_test_user@example.com"
    print(f"[PASS] GET /api/auth/me -> 200 (user={me_data['email']}, biz={me_data['business_name']})")

    # 3. Dashboard Metrics (Authenticated)
    r = client.get("/api/dashboard/metrics", headers=auth_header)
    assert r.status_code == 200, f"/dashboard/metrics failed: {r.text}"
    dash_data = r.json()
    assert "kpis" in dash_data, "kpis missing in dashboard metrics"
    assert "capacity" in dash_data, "capacity missing in dashboard metrics"
    assert "recent_anomalies" in dash_data, "recent_anomalies missing"
    print(f"[PASS] GET /api/dashboard/metrics -> 200 (latency={dash_data['kpis']['response_time_ms']}ms, cache_hit={dash_data['cache_hit']})")

    # 3b. Dashboard Metrics Cache Hit check
    r2 = client.get("/api/dashboard/metrics", headers=auth_header)
    assert r2.status_code == 200
    assert r2.json()["cache_hit"] is True, "Expected cache hit on 2nd request"
    print("[PASS] GET /api/dashboard/metrics (2nd call) -> 200 (cache_hit=True)")

    # 4. Telemetry Ingestion
    event_payload = {
        "event_type": "request",
        "response_time_ms": 95.4,
        "status_code": 200,
        "orders_count": 2,
        "endpoint": "/api/products",
        "idempotency_key": f"test-key-{uuid.uuid4().hex[:8]}",
    }
    r = client.post("/api/ingestion/events", json=event_payload, headers=auth_header)
    assert r.status_code == 202, f"Ingest failed: {r.text}"
    print("[PASS] POST /api/ingestion/events -> 202")

    # 5. Ingestion Service Batch Processing (In-Memory Buffer)
    import asyncio
    svc = IngestionService(stream_key=settings.REDIS_STREAM_KEY)
    processed = asyncio.run(svc.process_batch(batch_size=10))
    print(f"[PASS] IngestionService.process_batch -> drained {processed} in-memory events cleanly")

    # 6. ML Scheduled Jobs
    asyncio.run(run_anomaly_detection_job())
    print("[PASS] run_anomaly_detection_job() completed cleanly")

    asyncio.run(run_forecasting_job())
    print("[PASS] run_forecasting_job() completed cleanly")

    asyncio.run(run_retention_enforcement_job())
    print("[PASS] run_retention_enforcement_job() completed cleanly")

    # 7. Logout
    r = client.post("/api/auth/logout", headers=auth_header)
    assert r.status_code == 200, f"Logout failed: {r.text}"
    print("[PASS] POST /api/auth/logout -> 200 (token revoked in Redis memory blacklist)")

    # 8. Confirm revoked token is rejected
    r = client.get("/api/auth/me", headers=auth_header)
    assert r.status_code == 401, f"Expected 401 for revoked token, got {r.status_code}"
    print("[PASS] GET /api/auth/me with revoked token -> 401 Unauthorized")

    print("\nALL OFFLINE DECOUPLED SYSTEM TESTS PASSED 100%!")

if __name__ == "__main__":
    test_all()
