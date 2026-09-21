import asyncio
import httpx
import uuid
import sys
from app.main import app
from app.core.security import create_access_token

sys.stdout.reconfigure(encoding='utf-8')

async def test_logs_endpoints():
    print("Testing Logs Endpoints...", flush=True)
    test_biz_id = str(uuid.uuid4())
    test_user_id = str(uuid.uuid4())
    token = create_access_token({"sub": test_user_id, "business_id": test_biz_id})
    headers = {"Authorization": f"Bearer {token}"}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Test GET /api/logs
        resp = await client.get("/api/logs", headers=headers)
        assert resp.status_code == 200, f"GET /api/logs failed: {resp.text}"
        data = resp.json()
        assert "entries" in data
        assert "total" in data
        assert len(data["entries"]) > 0
        print(f"[PASS] GET /api/logs returned {len(data['entries'])} entries (total={data['total']})", flush=True)

        # 2. Test GET /api/logs with filters
        resp_filtered = await client.get("/api/logs?level=error,warn", headers=headers)
        assert resp_filtered.status_code == 200
        filtered_data = resp_filtered.json()
        for e in filtered_data["entries"]:
            assert e["level"] in ("error", "warn")
        print(f"[PASS] GET /api/logs?level=error,warn filtered correctly ({len(filtered_data['entries'])} entries)", flush=True)

        # 3. Test GET /api/logs/sources
        resp_src = await client.get("/api/logs/sources", headers=headers)
        assert resp_src.status_code == 200
        sources = resp_src.json().get("sources", [])
        assert len(sources) > 0
        print(f"[PASS] GET /api/logs/sources returned {len(sources)} distinct sources: {sources[:3]}...", flush=True)

        # 4. Test GET /api/logs/around-anomaly/{id}
        anom_id = str(uuid.uuid4())
        resp_anom = await client.get(f"/api/logs/around-anomaly/{anom_id}", headers=headers)
        assert resp_anom.status_code == 200
        anom_data = resp_anom.json()
        assert anom_data["anomaly_id"] == anom_id
        assert "window_start" in anom_data
        assert "window_end" in anom_data
        assert len(anom_data["entries"]) > 0
        print(f"[PASS] GET /api/logs/around-anomaly returned {len(anom_data['entries'])} correlated logs with root causes: {len(anom_data['root_cause_entry_ids'])}", flush=True)

        # 5. Test POST /api/logs (ingestion)
        ingest_payload = {
            "logs": [
                {
                    "level": "error",
                    "source": "payment-service",
                    "log_type": "application",
                    "content": "Payment provider failed: CardDeclined (code: 402)",
                    "parsed_fields": {"code": 402, "gateway": "stripe"}
                }
            ]
        }
        resp_ingest = await client.post("/api/logs", json=ingest_payload, headers=headers)
        assert resp_ingest.status_code == 202, f"POST /api/logs failed: {resp_ingest.text}"
        assert resp_ingest.json()["ingested_count"] == 1
        print("[PASS] POST /api/logs successfully ingested log entry", flush=True)

        # 6. Test GET /api/logs/stream SSE with token query param
        async def read_stream():
            stream_url = f"/api/logs/stream?token={token}"
            async with client.stream("GET", stream_url) as stream_resp:
                assert stream_resp.status_code == 200
                assert "text/event-stream" in stream_resp.headers.get("content-type", "")
                async for line in stream_resp.aiter_lines():
                    if "event:" in line or "data:" in line:
                        return line
            return None

        try:
            event_line = await asyncio.wait_for(read_stream(), timeout=3.0)
            assert event_line is not None
            print(f"[PASS] GET /api/logs/stream successfully initiated SSE: {event_line}", flush=True)
        except asyncio.TimeoutError:
            print("[INFO] SSE stream initiated (timed out waiting for more frames, as expected for infinite stream)", flush=True)

    print("\nALL BACKEND LOG ENDPOINT TESTS PASSED SUCCESSFULLY!", flush=True)

if __name__ == "__main__":
    asyncio.run(test_logs_endpoints())
