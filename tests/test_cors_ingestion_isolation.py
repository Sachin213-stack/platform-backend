import asyncio
import httpx
from app.main import app
from app.core.config import settings

async def test_cors_isolation():
    transport = httpx.ASGITransport(app=app)
    external_origin = "http://127.0.0.1:8088"
    malicious_origin = "https://evil-hacker.com"
    allowed_origin = "http://localhost:5173"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Acquire valid API key and tenant business ID
        res_auth = await client.post("/api/auth/demo")
        assert res_auth.status_code == 200
        auth_data = res_auth.json()
        token = auth_data["access_token"]
        biz_id = auth_data["business_id"]

        res_keys = await client.get("/api/users/api-keys", headers={"Authorization": f"Bearer {token}"})
        api_key = res_keys.json()[0]["api_key"]

        # 2. Test preflight OPTIONS on /api/ingestion/events from external origin
        res_opt_ingest = await client.options(
            "/api/ingestion/events",
            headers={
                "Origin": external_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, X-API-Key",
            },
        )
        assert res_opt_ingest.status_code == 200, f"Expected 200 on ingestion OPTIONS, got {res_opt_ingest.status_code}"
        assert res_opt_ingest.headers.get("access-control-allow-origin") == external_origin, "Missing CORS allow-origin on ingestion preflight"
        assert "x-api-key" in res_opt_ingest.headers.get("access-control-allow-headers", "").lower()
        print(f"[PASS] Cross-origin preflight to /api/ingestion/events allowed for {external_origin}")

        # 3. Test actual POST to /api/ingestion/events from external origin with valid API key
        res_post_ingest = await client.post(
            "/api/ingestion/events",
            headers={
                "Origin": external_origin,
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "business_id": biz_id,
                "event_type": "pageview",
                "endpoint": "/home",
                "response_time_ms": 35.0,
                "status_code": 200,
            },
        )
        assert res_post_ingest.status_code == 202, f"Expected 202 on ingestion, got {res_post_ingest.status_code}"
        assert res_post_ingest.headers.get("access-control-allow-origin") == external_origin
        print(f"[PASS] Cross-origin POST to /api/ingestion/events allowed for {external_origin}")

        # 4. Verify CORS IS NOT relaxed on other routes (e.g. /api/dashboard/metrics or /api/auth/me)
        res_other_opt = await client.options(
            "/api/dashboard/metrics",
            headers={
                "Origin": external_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        # Global CORSMiddleware does not allow external_origin for /dashboard/metrics
        allow_origin = res_other_opt.headers.get("access-control-allow-origin")
        assert allow_origin != external_origin and allow_origin != "*", (
            f"SECURITY VIOLATION: Non-ingestion route leaked CORS to external origin: {allow_origin}"
        )
        print(f"[PASS] Protected route /api/dashboard/metrics correctly BLOCKS cross-origin from {external_origin}")

        res_auth_opt = await client.options(
            "/api/auth/me",
            headers={
                "Origin": malicious_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        allow_origin_auth = res_auth_opt.headers.get("access-control-allow-origin")
        assert allow_origin_auth != malicious_origin and allow_origin_auth != "*", (
            f"SECURITY VIOLATION: Auth route leaked CORS to malicious origin: {allow_origin_auth}"
        )
        print(f"[PASS] Auth route /api/auth/me correctly BLOCKS cross-origin from {malicious_origin}")

        # 5. Confirm legitimate internal origin is still allowed by CORSMiddleware
        res_legit = await client.options(
            "/api/dashboard/metrics",
            headers={
                "Origin": allowed_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert res_legit.headers.get("access-control-allow-origin") == allowed_origin
        print(f"[PASS] Standard frontend origin {allowed_origin} successfully allowed via global CORSMiddleware")

        print("\nALL PHASE 4 CORS ISOLATION CHECKS PASSED 100%!")

if __name__ == "__main__":
    asyncio.run(test_cors_isolation())
