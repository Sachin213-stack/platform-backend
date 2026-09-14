"""
Unit and integration test validating dual route resolution:
Confirms both /api/... and /api/v1/... routes resolve properly and return HTTP 200.
"""

import asyncio
import httpx
from app.main import app


async def run_route_prefix_tests():
    print("=" * 80)
    print("RUNNING ROUTE PREFIX ALIAS VALIDATION TEST (/api vs /api/v1)")
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

        # 2. Generate demo token
        res_auth = await client.post("/api/auth/demo")
        assert res_auth.status_code == 200, f"Expected 200 on /api/auth/demo, got {res_auth.status_code}"
        token = res_auth.json().get("access_token")
        assert token, "No access_token returned from demo login"
        headers = {"Authorization": f"Bearer {token}"}
        print("[PASS] POST /api/auth/demo -> 200 (acquired demo token)")

        # 3. Test /api/dashboard/metrics (legacy / non-prefixed)
        res_legacy_dash = await client.get("/api/dashboard/metrics", headers=headers)
        assert res_legacy_dash.status_code == 200, f"Expected 200 on /api/dashboard/metrics, got {res_legacy_dash.status_code}"
        data_legacy = res_legacy_dash.json()
        assert "kpis" in data_legacy, "Missing 'kpis' in /api/dashboard/metrics response"
        assert "capacity" in data_legacy, "Missing 'capacity' in /api/dashboard/metrics response"
        assert "recent_anomalies" in data_legacy, "Missing 'recent_anomalies' in /api/dashboard/metrics response"
        print(f"[PASS] GET /api/dashboard/metrics -> {res_legacy_dash.status_code} (kpis={list(data_legacy['kpis'].keys())})")

        # 4. Test /api/v1/dashboard/metrics (versioned alias)
        res_v1_dash = await client.get("/api/v1/dashboard/metrics", headers=headers)
        assert res_v1_dash.status_code == 200, f"Expected 200 on /api/v1/dashboard/metrics, got {res_v1_dash.status_code}"
        data_v1 = res_v1_dash.json()
        assert "kpis" in data_v1, "Missing 'kpis' in /api/v1/dashboard/metrics response"
        assert "capacity" in data_v1, "Missing 'capacity' in /api/v1/dashboard/metrics response"
        assert "recent_anomalies" in data_v1, "Missing 'recent_anomalies' in /api/v1/dashboard/metrics response"
        print(f"[PASS] GET /api/v1/dashboard/metrics -> {res_v1_dash.status_code} (kpis={list(data_v1['kpis'].keys())})")

        # 5. Verify response schemas match
        assert set(data_legacy.keys()) == set(data_v1.keys()), "Schema mismatch between /api and /api/v1 responses"
        assert set(data_legacy["kpis"].keys()) == set(data_v1["kpis"].keys()), "KPI fields mismatch between prefixes"
        print("[PASS] Response schema equality between /api and /api/v1 confirmed")

    print("\nALL ROUTE PREFIX ALIAS TESTS PASSED SUCCESSFULLY (200 OK ON BOTH)!\n")


if __name__ == "__main__":
    asyncio.run(run_route_prefix_tests())
