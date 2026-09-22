import asyncio
import httpx
from app.main import app

async def test_api_key_flow():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Login with demo to get auth token
        res_auth = await client.post("/api/auth/demo")
        assert res_auth.status_code == 200, f"Demo login failed: {res_auth.text}"
        auth_data = res_auth.json()
        token = auth_data["access_token"]
        biz_id = auth_data["business_id"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Get or create API key
        res_keys = await client.get("/api/users/api-keys", headers=headers)
        assert res_keys.status_code == 200, f"Get keys failed: {res_keys.text}"
        keys = res_keys.json()
        assert len(keys) > 0, "No API keys returned"
        api_key = keys[0]["api_key"]
        assert api_key, "API key value is empty"
        print(f"[PASS] Retrieved API Key: {keys[0]['key_prefix']} (key={api_key[:16]}...) for biz={biz_id}")

        # 3. Create another named API key
        res_create = await client.post("/api/users/api-keys", headers=headers, json={"name": "E2E Snippet Test Key"})
        assert res_create.status_code == 201, f"Create key failed: {res_create.text}"
        new_key_data = res_create.json()
        created_key = new_key_data["api_key"]
        assert created_key.startswith("sk_live_")
        print(f"[PASS] Created new API Key: {created_key[:16]}...")

        # 4. Ingest event using X-API-Key header (WITHOUT Bearer token)
        event_payload = {
            "business_id": biz_id,
            "event_type": "pageview",
            "endpoint": "/products/summer-collection",
            "response_time_ms": 52.3,
            "status_code": 200,
        }
        res_ingest = await client.post(
            "/api/ingestion/events",
            headers={"X-API-Key": created_key, "Content-Type": "application/json"},
            json=event_payload,
        )
        assert res_ingest.status_code == 202, f"Ingestion with X-API-Key failed: {res_ingest.status_code} - {res_ingest.text}"
        ingest_data = res_ingest.json()
        assert ingest_data["status"] == "accepted", f"Expected accepted, got {ingest_data}"
        print(f"[PASS] Telemetry event ingested successfully via X-API-Key: event_id={ingest_data['event_id']}")

        # 5. Ingest event with wrong business_id -> must be rejected with 403 (cross-tenant spoof check)
        spoofed_payload = {
            "business_id": "22222222-2222-2222-2222-222222222222",
            "event_type": "pageview",
            "endpoint": "/spoof",
            "response_time_ms": 10.0,
            "status_code": 200,
        }
        res_spoof = await client.post(
            "/api/ingestion/events",
            headers={"X-API-Key": created_key, "Content-Type": "application/json"},
            json=spoofed_payload,
        )
        assert res_spoof.status_code == 403, f"Expected 403 on cross-tenant spoof, got {res_spoof.status_code}"
        print(f"[PASS] Cross-tenant spoof attempt properly rejected with 403 Forbidden")

        print("\nALL PHASE 3 API-KEY VERIFICATIONS PASSED!")

if __name__ == "__main__":
    asyncio.run(test_api_key_flow())
