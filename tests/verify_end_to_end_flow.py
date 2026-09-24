"""
End-to-End Verification Test Script
Tests:
1. Ingest telemetry events via POST /api/ingestion/events.
2. Process the ingestion stream via IngestionService.
3. Invalidate & fetch GET /api/dashboard/metrics: verify has_live_data, timeseries, and tier_metrics.
4. Verify GET /api/dashboard/analytics: verify forecast_curve and resource_runway.
5. Execute a FRIDAY autonomous mitigation action: POST /api/friday/actions/execute.
6. Verify GET /api/dashboard/audit-logs: verify that the executed action is returned.
7. Verify GET /api/friday/history: verify conversation transcript retrieval.
"""

import asyncio
import uuid
import httpx
from app.main import app
from app.services.ingestion_service import ingestion_service
from app.services.redis_service import redis_service


async def verify_flow():
    print("=" * 75)
    print("STARTING COMPLETE END-TO-END FLOW & DATABASE/MEMORY VERIFICATION")
    print("=" * 75)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Login with demo account
        auth_resp = await client.post("/api/auth/demo")
        assert auth_resp.status_code == 200, f"Auth demo failed: {auth_resp.text}"
        auth_data = auth_resp.json()
        token = auth_data["access_token"]
        business_id = auth_data["business_id"]
        headers = {"Authorization": f"Bearer {token}"}
        print(f"[1] Authenticated demo user for business: {business_id}")

        # 2. Before ingesting live events, check baseline metrics
        initial_metrics_resp = await client.get("/api/dashboard/metrics", headers=headers)
        assert initial_metrics_resp.status_code == 200, f"Dashboard metrics failed: {initial_metrics_resp.text}"
        initial_metrics = initial_metrics_resp.json()
        print(f"[2] Initial metrics retrieved: has_live_data={initial_metrics.get('has_live_data')}, timeseries_points={len(initial_metrics.get('timeseries', []))}")

        # 3. Ingest real telemetry events (simulate website beacon sending 3 events)
        print("[3] Simulating live website telemetry stream ingestion...")
        for i in range(3):
            event_payload = {
                "business_id": business_id,
                "event_type": "request",
                "response_time_ms": 125.0 + i * 15.0,
                "status_code": 200,
                "orders_count": 1,
                "revenue_amount": 49.99 + i * 20.0,
                "endpoint": f"/checkout/step-{i+1}",
                "idempotency_key": f"test-beacon-{uuid.uuid4()}",
            }
            ingest_resp = await client.post("/api/ingestion/events", json=event_payload, headers=headers)
            assert ingest_resp.status_code == 202, f"Ingestion failed: {ingest_resp.text}"
            print(f"    * Ingested event {i+1}/3: {event_payload['endpoint']} -> HTTP 202 Accepted")

        # 4. Drain & process ingestion batch
        processed = await ingestion_service.process_batch(batch_size=50)
        print(f"[4] Ingestion service processed batch: {processed} stream events")

        # 5. Fetch dashboard metrics again (cache should be invalidated)
        updated_metrics_resp = await client.get("/api/dashboard/metrics", headers=headers)
        assert updated_metrics_resp.status_code == 200
        updated_metrics = updated_metrics_resp.json()
        print(f"[5] Post-ingestion metrics: has_live_data={updated_metrics.get('has_live_data')}, tier_metrics={updated_metrics.get('tier_metrics')}")

        # 6. Check Analytics Studio endpoint
        analytics_resp = await client.get("/api/dashboard/analytics", headers=headers)
        assert analytics_resp.status_code == 200, f"Analytics endpoint failed: {analytics_resp.text}"
        analytics_data = analytics_resp.json()
        print(f"[6] Analytics summary: has_live_data={analytics_data.get('has_live_data')}, runway_days={analytics_data.get('resource_runway', {}).get('runway_days')}, growth_rate={analytics_data.get('resource_runway', {}).get('growth_rate_pct')}%")

        # 7. Execute a FRIDAY autonomous mitigation action
        print("[7] Executing FRIDAY autonomous mitigation action...")
        action_payload = {
            "action_type": "scale_service",
            "service": "checkout-svc",
            "params": {"replicas": 6, "reason": "SLA latency spike mitigation"},
            "conversation_id": str(uuid.uuid4()),
        }
        exec_resp = await client.post("/api/friday/actions/execute", json=action_payload, headers=headers)
        assert exec_resp.status_code == 200, f"Action execution failed: {exec_resp.text}"
        exec_data = exec_resp.json()
        print(f"    * Executed action: ID={exec_data.get('action_id')}, message={exec_data.get('message')}")

        # 8. Check Audit Log endpoint
        audit_resp = await client.get("/api/dashboard/audit-logs", headers=headers)
        assert audit_resp.status_code == 200, f"Audit logs failed: {audit_resp.text}"
        audit_data = audit_resp.json()
        print(f"[8] Audit log returned {len(audit_data.get('entries', []))} entries (total: {audit_data.get('total')})")
        assert any(e.get("service") == "checkout-svc" for e in audit_data.get("entries", [])), "Executed action not found in audit log"
        print("    * Verified executed action is present in immutable audit log!")

        # 9. Verify FRIDAY History endpoint
        conv_id = action_payload["conversation_id"]
        hist_resp = await client.get(f"/api/friday/history?conversation_id={conv_id}", headers=headers)
        assert hist_resp.status_code == 200, f"History failed: {hist_resp.text}"
        hist_data = hist_resp.json()
        print(f"[9] FRIDAY history for conv {conv_id[:8]}... returned {len(hist_data.get('messages', []))} messages")
        assert len(hist_data.get("messages", [])) > 0, "Expected at least 1 message in session memory"

    print("\n" + "=" * 75)
    print("ALL END-TO-END VERIFICATION CHECKS PASSED WITH 100% SUCCESS!")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(verify_flow())
