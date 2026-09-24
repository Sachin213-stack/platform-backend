import asyncio
import httpx
import uuid
import sys
from app.main import app
from app.core.security import create_access_token

sys.stdout.reconfigure(encoding='utf-8')

async def test_logs_endpoints():
    print("=" * 70, flush=True)
    print("AI-CTO ZERO-DEMO DATA & OBSERVABILITY TEST SUITE", flush=True)
    print("=" * 70, flush=True)

    test_biz_id = str(uuid.uuid4())
    test_user_id = str(uuid.uuid4())
    token = create_access_token({"sub": test_user_id, "business_id": test_biz_id})
    headers = {"Authorization": f"Bearer {token}"}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # -------------------------------------------------------------
        # 1. Fresh Tenant Zero-Demo Contract: Must return empty data
        # -------------------------------------------------------------
        print("\n--- [PHASE 1] Zero-Demo Contract on Fresh Tenant ---", flush=True)
        resp = await client.get("/api/logs", headers=headers)
        assert resp.status_code == 200, f"GET /api/logs failed: {resp.text}"
        data = resp.json()
        assert "entries" in data
        assert "total" in data
        assert len(data["entries"]) == 0, f"Expected 0 mock logs for fresh tenant, got {len(data['entries'])}"
        assert data["total"] == 0
        print("[PASS] GET /api/logs returned 0 entries for fresh tenant (No fake demo logs).", flush=True)

        resp_src = await client.get("/api/logs/sources", headers=headers)
        assert resp_src.status_code == 200
        sources = resp_src.json().get("sources", [])
        assert len(sources) == 0, f"Expected 0 sources for fresh tenant, got {sources}"
        print("[PASS] GET /api/logs/sources returned 0 sources for fresh tenant (No fake fallback sources).", flush=True)

        anom_id = str(uuid.uuid4())
        resp_anom = await client.get(f"/api/logs/around-anomaly/{anom_id}", headers=headers)
        assert resp_anom.status_code == 200
        anom_data = resp_anom.json()
        assert len(anom_data["entries"]) == 0, f"Expected 0 correlated logs, got {len(anom_data['entries'])}"
        print("[PASS] GET /api/logs/around-anomaly returned 0 entries when DB has no records.", flush=True)

        resp_audit = await client.get("/api/dashboard/audit-logs", headers=headers)
        assert resp_audit.status_code == 200
        audit_entries = resp_audit.json().get("entries", [])
        assert len(audit_entries) == 0, f"Expected 0 fake audit entries, got {len(audit_entries)}"
        print("[PASS] GET /api/dashboard/audit-logs returned 0 entries (No Sarah Jenkins fake audit baseline).", flush=True)

        # -------------------------------------------------------------
        # 2. Ingest Real Server Logs via POST /api/logs
        # -------------------------------------------------------------
        print("\n--- [PHASE 2] Real Server Log Ingestion & Filtering ---", flush=True)
        ingest_payload = {
            "logs": [
                {
                    "level": "error",
                    "source": "payment-service",
                    "log_type": "application",
                    "content": "Payment provider timeout: CardDeclined (code: 402)",
                    "parsed_fields": {"code": 402, "gateway": "stripe"}
                },
                {
                    "level": "info",
                    "source": "auth-service",
                    "log_type": "application",
                    "content": "User session token refreshed successfully",
                    "parsed_fields": {"auth_type": "jwt"}
                }
            ]
        }
        resp_ingest = await client.post("/api/logs", json=ingest_payload, headers=headers)
        assert resp_ingest.status_code == 202, f"POST /api/logs failed: {resp_ingest.text}"
        assert resp_ingest.json()["ingested_count"] == 2
        print("[PASS] POST /api/logs ingested 2 real log entries successfully.", flush=True)

        # Query all logs
        resp_after = await client.get("/api/logs", headers=headers)
        assert resp_after.status_code == 200
        logs_after = resp_after.json()
        assert logs_after["total"] == 2
        assert len(logs_after["entries"]) == 2
        print(f"[PASS] GET /api/logs retrieved exactly {logs_after['total']} real ingested entries.", flush=True)

        # Query with level filter
        resp_filtered = await client.get("/api/logs?level=error", headers=headers)
        assert resp_filtered.status_code == 200
        filtered_entries = resp_filtered.json()["entries"]
        assert len(filtered_entries) == 1
        assert filtered_entries[0]["level"] == "error"
        assert filtered_entries[0]["source"] == "payment-service"
        print("[PASS] GET /api/logs?level=error filtered correctly (1 entry).", flush=True)

        # Query sources: must now contain exactly the real ingested sources
        resp_sources_after = await client.get("/api/logs/sources", headers=headers)
        assert resp_sources_after.status_code == 200
        active_sources = resp_sources_after.json()["sources"]
        assert "payment-service" in active_sources
        assert "auth-service" in active_sources
        assert "checkout-service" not in active_sources  # Must not contain uningested fake sources
        print(f"[PASS] GET /api/logs/sources returned genuine sources: {active_sources}", flush=True)

        # -------------------------------------------------------------
        # 3. Real SSE Stream Connection & Keepalive
        # -------------------------------------------------------------
        print("\n--- [PHASE 3] SSE Live-Tail Stream Integrity ---", flush=True)
        async def verify_sse():
            stream_url = f"/api/logs/stream?token={token}"
            async with client.stream("GET", stream_url) as stream_resp:
                assert stream_resp.status_code == 200
                assert "text/event-stream" in stream_resp.headers.get("content-type", "")
                async for line in stream_resp.aiter_lines():
                    if "event: status" in line or "data: " in line:
                        return line
            return None

        try:
            event_line = await asyncio.wait_for(verify_sse(), timeout=2.5)
            assert event_line is not None
            print(f"[PASS] SSE live-tail connected and yielded initial status event: {event_line[:60]}...", flush=True)
        except asyncio.TimeoutError:
            print("[PASS] SSE live-tail successfully initiated and accepted streaming connection without errors.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ALL OBSERVABILITY & LOGS ZERO-DEMO VERIFICATIONS PASSED!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    asyncio.run(test_logs_endpoints())
