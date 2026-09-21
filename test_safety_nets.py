import unittest
import uuid
import json
import asyncio
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.security import create_access_token
from app.services.redis_service import redis_service
from app.services.ingestion_service import ingestion_service
from app.services.llm_adapter import llm_adapter


class TestSafetyNets(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.biz_id = str(uuid.uuid4())
        self.admin_token = create_access_token({
            "sub": str(uuid.uuid4()),
            "business_id": self.biz_id,
            "role": "admin",
        })
        self.viewer_token = create_access_token({
            "sub": str(uuid.uuid4()),
            "business_id": self.biz_id,
            "role": "viewer",
        })

    # =========================================================================
    # 1. Clamping Flag & Raw Observed Metric Preservation Tests
    # =========================================================================
    def test_clamping_flag_preserves_raw_values(self):
        """Verify extreme metrics are bounded safely while preserving raw observed data in metadata."""
        captured_payloads = []

        async def mock_add_to_stream(stream_key, data):
            captured_payloads.append(data)
            return "1720000000000-0"

        with patch("app.services.redis_service.redis_service.add_to_stream", side_effect=mock_add_to_stream):
            resp = self.client.post(
                "/api/v1/ingestion/events",
                headers={"Authorization": f"Bearer {self.admin_token}"},
                json={
                    "business_id": self.biz_id,
                    "event_type": "api_request",
                    "response_time_ms": 185000.0,  # 185 seconds hung query
                    "cpu_usage_pct": 140.0,        # Oversaturated reading
                    "endpoint": "/api/v1/heavy-query",
                },
            )
            self.assertEqual(resp.status_code, 202)
            self.assertEqual(len(captured_payloads), 1)

            payload = captured_payloads[0]
            # Verify numerical boundedness for downstream models
            self.assertEqual(payload["response_time_ms"], 60000.0)
            self.assertEqual(payload["cpu_usage_pct"], 100.0)

            # Verify Clamping Flag preservation
            clamping = payload["payload_metadata"].get("clamping", {})
            self.assertTrue(clamping.get("is_clamped"))
            self.assertEqual(clamping.get("raw_response_time_ms"), 185000.0)
            self.assertEqual(clamping.get("raw_cpu_usage_pct"), 140.0)
            self.assertIn("safe operational numerical boundaries", clamping.get("reason", ""))

    # =========================================================================
    # 2. Dead-Letter Queue (DLQ) Tests
    # =========================================================================
    def test_dlq_diverts_poison_pill_stream_entry(self):
        """Verify malformed stream entry is diverted to DLQ without crashing the batch consumer."""
        dlq_entries = []

        async def mock_route_to_dlq(entry_id, fields, reason):
            dlq_entries.append({"entry_id": entry_id, "reason": reason, "fields": fields})

        # Clear in-memory streams
        redis_service._memory_streams[settings.REDIS_STREAM_KEY] = [
            ("valid-1", {
                "id": str(uuid.uuid4()),
                "business_id": self.biz_id,
                "event_type": "request",
                "response_time_ms": "120.0",
                "status_code": "200",
            }),
            ("poison-pill-2", {
                "id": "corrupted-entry-without-business-id",
                # missing business_id
                "status_code": "garbage",
            }),
        ]

        with patch.object(ingestion_service, "route_to_dlq", side_effect=mock_route_to_dlq):
            processed_count = asyncio.run(ingestion_service.process_batch(batch_size=10))

            self.assertEqual(processed_count, 2)
            self.assertEqual(len(dlq_entries), 1)
            self.assertEqual(dlq_entries[0]["entry_id"], "poison-pill-2")
            self.assertIn("Missing or invalid business_id", dlq_entries[0]["reason"])

    def test_dlq_endpoint_rbac(self):
        """Verify GET /api/v1/ingestion/dlq requires admin/owner privileges."""
        # Viewer forbidden
        resp = self.client.get(
            "/api/v1/ingestion/dlq",
            headers={"Authorization": f"Bearer {self.viewer_token}"},
        )
        self.assertEqual(resp.status_code, 403)

        # Admin authorized
        resp = self.client.get(
            "/api/v1/ingestion/dlq",
            headers={"Authorization": f"Bearer {self.admin_token}"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["dlq_stream"], settings.REDIS_DLQ_STREAM_KEY)
        self.assertIn("total_quarantined", data)

    # =========================================================================
    # 3. Multi-Metric Correlation (Flash Sale Protection) Tests
    # =========================================================================
    def test_multi_metric_correlation_suppresses_throttling_during_flash_sale(self):
        """Verify propose_mitigation_action overrides throttle_rate_limits during healthy flash surge."""
        async def mock_context(biz_id):
            return {
                "business_name": "Mega Store",
                "avg_response_time_ms": 350.0,
                "error_rate_pct": 0.05,  # Very low error rate (0.05%)
                "orders_per_min": 18.5,  # High order volume (18.5 orders/min)
                "cpu_usage_pct": 82.0,
                "memory_usage_pct": 74.0,
                "queue_depth": 12,
                "active_anomalies": [],
                "alert_thresholds": [],
                "crash_risk_pct": 14.0,
            }

        with patch.object(llm_adapter, "get_live_business_context", side_effect=mock_context):
            result = asyncio.run(llm_adapter.execute_tool(
                "propose_mitigation_action",
                {
                    "action_type": "throttle_rate_limits",
                    "service": "checkout-v2",
                    "params": {"limit_per_sec": 100},
                    "rationale": "High latency detected on checkout",
                },
                self.biz_id,
            ))

            # Verify action was automatically overridden to scale_service to protect revenue
            self.assertEqual(result["action_type"], "scale_service")
            self.assertGreaterEqual(result["params"]["replicas"], 6)
            self.assertIn("Multi-metric correlation", result["rationale"])
            self.assertIn("protect customer revenue", result["rationale"])


if __name__ == "__main__":
    unittest.main()
