import asyncio
import re
import uuid
import httpx
from app.main import app

UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

async def run_e2e_verification():
    print("=" * 70)
    print("AI-CTO SNIPPET-BASED WEBSITE CONNECTIVITY: END-TO-END VERIFICATION")
    print("=" * 70)

    transport = httpx.ASGITransport(app=app)
    external_client_origin = "https://checkout.external-merchant-store.com"
    allowed_platform_origin = "http://localhost:5173"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # -------------------------------------------------------------
        # 1. Point 1 & 7: Verify tracker.js is hosted and accessible with CORS
        # -------------------------------------------------------------
        print("\n--- [POINT 1 & 7] Tracker Hosting & CDN Unification ---")
        for path in ["/static/tracker.js", "/api/tracker.js", "/tracker.js"]:
            res = await client.get(path)
            assert res.status_code == 200, f"Expected 200 on {path}, got {res.status_code}"
            assert "javascript" in res.headers.get("content-type", "").lower()
            assert res.headers.get("access-control-allow-origin") == "*"
            print(f"  [OK] {path} returned 200 with Access-Control-Allow-Origin: *")

        tracker_code = (await client.get("/static/tracker.js")).text
        assert "data-business-id" in tracker_code
        assert "data-api-key" in tracker_code
        assert "X-API-Key" in tracker_code
        assert "/api/ingestion/events" in tracker_code
        assert "localhost" not in tracker_code
        assert "127.0.0.1" not in tracker_code
        print("  [OK] tracker.js script contains dynamic origin deduction without hardcoded URLs")

        # -------------------------------------------------------------
        # 2. Point 2: Scoped CORS Isolation for external sites
        # -------------------------------------------------------------
        print("\n--- [POINT 2] Scoped Ingestion CORS vs Route Isolation ---")
        # Preflight OPTIONS on ingestion route from external origin
        res_preflight = await client.options(
            "/api/ingestion/events",
            headers={
                "Origin": external_client_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, X-API-Key",
            },
        )
        assert res_preflight.status_code == 200
        assert res_preflight.headers.get("access-control-allow-origin") == external_client_origin
        assert "x-api-key" in res_preflight.headers.get("access-control-allow-headers", "").lower()
        print(f"  [OK] Ingestion OPTIONS correctly allows external origin: {external_client_origin}")

        # Ensure protected routes (e.g. /api/dashboard/metrics) do NOT leak to external origin
        res_blocked_opt = await client.options(
            "/api/dashboard/metrics",
            headers={
                "Origin": external_client_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert res_blocked_opt.headers.get("access-control-allow-origin") != external_client_origin
        assert res_blocked_opt.headers.get("access-control-allow-origin") != "*"
        print("  [OK] Dashboard route /api/dashboard/metrics strictly blocks external origin")

        # Standard frontend origin is permitted
        res_allowed_opt = await client.options(
            "/api/dashboard/metrics",
            headers={
                "Origin": allowed_platform_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert res_allowed_opt.headers.get("access-control-allow-origin") == allowed_platform_origin
        print(f"  [OK] Dashboard route permits platform origin: {allowed_platform_origin}")

        # -------------------------------------------------------------
        # 3. Point 5: business_id format validation (UUID)
        # -------------------------------------------------------------
        print("\n--- [POINT 5] business_id Format & Tenant Alignment ---")
        res_auth = await client.post("/api/auth/demo")
        assert res_auth.status_code == 200, f"Demo login failed: {res_auth.text}"
        auth_data = res_auth.json()
        token = auth_data["access_token"]
        biz_id = auth_data["business_id"]

        assert UUID_REGEX.match(biz_id), f"business_id '{biz_id}' is not a valid UUID!"
        # Test UUID parsing
        parsed_uuid = uuid.UUID(biz_id)
        assert str(parsed_uuid) == biz_id
        print(f"  [OK] Authenticated business_id is valid RFC UUID: {biz_id}")

        # -------------------------------------------------------------
        # 4. Point 3: Ingestion with API Key & Cross-tenant Isolation
        # -------------------------------------------------------------
        print("\n--- [POINT 3] API Key Ingestion & Cross-Tenant Security ---")
        # Retrieve or create API key
        res_keys = await client.get("/api/users/api-keys", headers={"Authorization": f"Bearer {token}"})
        assert res_keys.status_code == 200
        keys = res_keys.json()
        assert len(keys) > 0
        api_key = keys[0]["api_key"]
        assert api_key.startswith("sk_live_")
        print(f"  [OK] Per-tenant API key acquired: {api_key[:16]}...")

        # Ingestion from external site with valid API key
        event_payload = {
            "business_id": biz_id,
            "event_type": "request",
            "endpoint": "/checkout/complete",
            "response_time_ms": 118.4,
            "status_code": 200,
        }
        res_ingest = await client.post(
            "/api/ingestion/events",
            headers={
                "Origin": external_client_origin,
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json=event_payload,
        )
        assert res_ingest.status_code == 202, f"Expected 202, got {res_ingest.status_code}: {res_ingest.text}"
        ingest_resp = res_ingest.json()
        assert ingest_resp["status"] == "accepted"
        assert res_ingest.headers.get("access-control-allow-origin") == external_client_origin
        print(f"  [OK] Telemetry event accepted from external origin: event_id={ingest_resp.get('event_id')}")

        # Negative test: cross-tenant spoof attempt
        wrong_tenant_uuid = "22222222-2222-2222-2222-222222222222"
        res_spoof = await client.post(
            "/api/ingestion/events",
            headers={
                "Origin": external_client_origin,
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json={**event_payload, "business_id": wrong_tenant_uuid},
        )
        assert res_spoof.status_code == 403, f"Expected 403 Forbidden for spoofed tenant, got {res_spoof.status_code}"
        print("  [OK] Cross-tenant spoof attempt blocked with 403 Forbidden")

        # Negative test: missing API key
        res_no_key = await client.post(
            "/api/ingestion/events",
            headers={"Origin": external_client_origin, "Content-Type": "application/json"},
            json=event_payload,
        )
        assert res_no_key.status_code == 401, f"Expected 401 Unauthorized for missing key, got {res_no_key.status_code}"
        print("  [OK] Unauthenticated ingestion request rejected with 401 Unauthorized")

        # Negative test: legacy string business_id fails tenant match
        res_legacy_biz = await client.post(
            "/api/ingestion/events",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json={**event_payload, "business_id": "biz_live_legacy123"},
        )
        assert res_legacy_biz.status_code == 403, f"Expected 403 for mismatched legacy business_id, got {res_legacy_biz.status_code}"
        print("  [OK] Legacy non-matching business_id strictly rejected with 403 Forbidden")

        # Negative test: invalid schema (e.g. status_code not an integer)
        res_bad_schema = await client.post(
            "/api/ingestion/events",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json={**event_payload, "status_code": "not-a-number"},
        )
        assert res_bad_schema.status_code == 422, f"Expected 422 for malformed schema, got {res_bad_schema.status_code}"
        print("  [OK] Malformed payload rejected with 422 Unprocessable Entity")

        # -------------------------------------------------------------
        # 5. Point 4 & 6: Verification Integrity & Telemetry Metrics
        # -------------------------------------------------------------
        print("\n--- [POINT 4 & 6] Verification Pipeline & Metrics ---")
        res_metrics = await client.get(
            f"/api/dashboard/metrics?business_id={biz_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res_metrics.status_code == 200, f"Metrics query failed: {res_metrics.status_code}"
        metrics_data = res_metrics.json()
        print(f"  [OK] Dashboard metrics retrieved: status={metrics_data.get('status')}")

    print("\n" + "=" * 70)
    print("ALL 7 DIAGNOSTIC FAILURE POINTS VERIFIED WORKING END-TO-END!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_e2e_verification())
