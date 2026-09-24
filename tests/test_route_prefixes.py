"""
Unit and integration test validating dual route resolution:
Confirms routes resolve properly and have complete schema parity between
/api/... and /api/v1/... for all mounted routers:
- Health router (/health)
- Auth router (/auth/demo, /auth/me)
- Dashboard router (/dashboard/metrics)
- Users router (/users/me)
- Ingestion router (/ingestion/events)
- Friday router (/friday/actions/execute)
"""

import asyncio
import uuid
import httpx
from app.main import app


async def run_route_prefix_tests():
    print("=" * 80)
    print("RUNNING EXTENDED ROUTE PREFIX ALIAS VALIDATION TEST (/api vs /api/v1)")
    print("=" * 80)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check resolution on both prefixes
        res_legacy_health = await client.get("/api/health")
        assert res_legacy_health.status_code == 200, f"Expected 200 on /api/health, got {res_legacy_health.status_code}"
        print(f"[PASS] GET /api/health -> {res_legacy_health.status_code}")

        res_v1_health = await client.get("/api/v1/health")
        assert res_v1_health.status_code == 200, f"Expected 200 on /api/v1/health, got {res_v1_health.status_code}"
        print(f"[PASS] GET /api/v1/health -> {res_v1_health.status_code}")
        assert set(res_legacy_health.json().keys()) == set(res_v1_health.json().keys()), "Health schema mismatch"

        # Also confirm root /health resolves
        res_root_health = await client.get("/health")
        assert res_root_health.status_code == 200, f"Expected 200 on /health, got {res_root_health.status_code}"
        print(f"[PASS] GET /health -> {res_root_health.status_code}")

        # 2. Auth - Generate demo token & test /auth/me
        res_auth_legacy = await client.post("/api/auth/demo")
        assert res_auth_legacy.status_code == 200, f"Expected 200 on /api/auth/demo, got {res_auth_legacy.status_code}"
        token = res_auth_legacy.json().get("access_token")
        assert token, "No access_token returned from demo login"
        headers = {"Authorization": f"Bearer {token}"}
        print("[PASS] POST /api/auth/demo -> 200 (acquired demo token)")

        res_auth_v1 = await client.post("/api/v1/auth/demo")
        assert res_auth_v1.status_code == 200, f"Expected 200 on /api/v1/auth/demo, got {res_auth_v1.status_code}"
        print("[PASS] POST /api/v1/auth/demo -> 200")
        assert set(res_auth_legacy.json().keys()) == set(res_auth_v1.json().keys()), "Auth demo schema mismatch"

        res_auth_me_legacy = await client.get("/api/auth/me", headers=headers)
        assert res_auth_me_legacy.status_code == 200
        res_auth_me_v1 = await client.get("/api/v1/auth/me", headers=headers)
        assert res_auth_me_v1.status_code == 200
        assert set(res_auth_me_legacy.json().keys()) == set(res_auth_me_v1.json().keys()), "Auth me schema mismatch"
        print("[PASS] GET /api/auth/me & /api/v1/auth/me -> 200 (schema equality confirmed)")

        # 3. Dashboard Metrics
        res_legacy_dash = await client.get("/api/dashboard/metrics", headers=headers)
        assert res_legacy_dash.status_code == 200, f"Expected 200 on /api/dashboard/metrics, got {res_legacy_dash.status_code}"
        data_legacy = res_legacy_dash.json()
        assert "kpis" in data_legacy, "Missing 'kpis' in /api/dashboard/metrics response"
        assert "capacity" in data_legacy, "Missing 'capacity' in /api/dashboard/metrics response"
        assert "recent_anomalies" in data_legacy, "Missing 'recent_anomalies' in /api/dashboard/metrics response"

        res_v1_dash = await client.get("/api/v1/dashboard/metrics", headers=headers)
        assert res_v1_dash.status_code == 200, f"Expected 200 on /api/v1/dashboard/metrics, got {res_v1_dash.status_code}"
        data_v1 = res_v1_dash.json()
        assert "kpis" in data_v1, "Missing 'kpis' in /api/v1/dashboard/metrics response"
        assert "capacity" in data_v1, "Missing 'capacity' in /api/v1/dashboard/metrics response"
        assert "recent_anomalies" in data_v1, "Missing 'recent_anomalies' in /api/v1/dashboard/metrics response"

        assert set(data_legacy.keys()) == set(data_v1.keys()), "Schema mismatch between /api and /api/v1 dashboard responses"
        assert set(data_legacy["kpis"].keys()) == set(data_v1["kpis"].keys()), "KPI fields mismatch between prefixes"
        print(f"[PASS] GET /api/dashboard/metrics & /api/v1/dashboard/metrics -> 200 (schema equality confirmed)")

        # 4. Users Router (/users/me)
        res_user_legacy = await client.get("/api/users/me", headers=headers)
        assert res_user_legacy.status_code == 200, f"Expected 200 on /api/users/me, got {res_user_legacy.status_code}"
        res_user_v1 = await client.get("/api/v1/users/me", headers=headers)
        assert res_user_v1.status_code == 200, f"Expected 200 on /api/v1/users/me, got {res_user_v1.status_code}"
        assert set(res_user_legacy.json().keys()) == set(res_user_v1.json().keys()), "Users /me schema mismatch"
        print(f"[PASS] GET /api/users/me & /api/v1/users/me -> 200 (schema equality confirmed)")

        # 5. Ingestion Router (/ingestion/events)
        ingest_payload = {
            "event_type": "request",
            "response_time_ms": 115.4,
            "status_code": 200,
            "orders_count": 1,
            "endpoint": "/checkout",
            "idempotency_key": str(uuid.uuid4()),
        }
        res_ingest_legacy = await client.post("/api/ingestion/events", json=ingest_payload, headers=headers)
        assert res_ingest_legacy.status_code in [200, 202], f"Expected 200/202 on /api/ingestion/events, got {res_ingest_legacy.status_code}"

        ingest_payload["idempotency_key"] = str(uuid.uuid4())
        res_ingest_v1 = await client.post("/api/v1/ingestion/events", json=ingest_payload, headers=headers)
        assert res_ingest_v1.status_code in [200, 202], f"Expected 200/202 on /api/v1/ingestion/events, got {res_ingest_v1.status_code}"
        assert res_ingest_legacy.status_code == res_ingest_v1.status_code, "Ingestion status code mismatch"
        assert set(res_ingest_legacy.json().keys()) == set(res_ingest_v1.json().keys()), "Ingestion schema mismatch"
        print(f"[PASS] POST /api/ingestion/events & /api/v1/ingestion/events -> {res_ingest_legacy.status_code} (schema equality confirmed)")

        # 6. Friday Router (/friday/actions/execute)
        action_payload = {
            "action_type": "scale_service",
            "service": "orders-service",
            "params": {"replicas": 4},
        }
        res_friday_legacy = await client.post("/api/friday/actions/execute", json=action_payload, headers=headers)
        assert res_friday_legacy.status_code == 200, f"Expected 200 on /api/friday/actions/execute, got {res_friday_legacy.status_code}"
        res_friday_v1 = await client.post("/api/v1/friday/actions/execute", json=action_payload, headers=headers)
        assert res_friday_v1.status_code == 200, f"Expected 200 on /api/v1/friday/actions/execute, got {res_friday_v1.status_code}"
        assert set(res_friday_legacy.json().keys()) == set(res_friday_v1.json().keys()), "Friday actions schema mismatch"
        print(f"[PASS] POST /api/friday/actions/execute & /api/v1/friday/actions/execute -> 200 (schema equality confirmed)")

    print("\nALL EXTENDED ROUTE PREFIX ALIAS TESTS PASSED (HEALTH, AUTH, DASHBOARD, USERS, INGESTION, FRIDAY)!\n")


if __name__ == "__main__":
    asyncio.run(run_route_prefix_tests())
