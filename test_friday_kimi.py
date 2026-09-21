import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
import json
import uuid
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.llm_adapter import (
    llm_adapter,
    KimiAuthenticationError,
    AllEndpointsExhaustedError,
    RateLimitError,
)

class TestFridayKimiIntegration(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        try:
            if hasattr(app.state, "ml_worker") and app.state.ml_worker:
                app.state.ml_worker.stop()
            if hasattr(app.state, "ingestion_worker") and app.state.ingestion_worker:
                app.state.ingestion_worker.stop()
        except Exception:
            pass

    def setUp(self):
        self.client = TestClient(app)
        from app.core.security import create_access_token
        self.token = create_access_token({
            "sub": str(uuid.uuid4()),
            "business_id": str(uuid.uuid4()),
            "role": "owner",
        })
        self.auth_headers = {"Authorization": f"Bearer {self.token}"}

    def test_missing_api_key_fails_loudly(self):
        """Verify that missing KIMI_API_KEY returns HTTP 503 with explicit error message, not mock data."""
        with patch.object(settings, "KIMI_API_KEY", ""):
            with patch.object(settings, "MOONSHOT_API_KEY", ""):
                with patch.object(settings, "NVIDIA_API_KEY", ""):
                    with patch.dict("os.environ", {"KIMI_API_KEY": "", "MOONSHOT_API_KEY": "", "NVIDIA_API_KEY": ""}, clear=False):
                        resp = self.client.post(
                            "/api/friday/chat",
                            json={"message": "What is our current crash risk?"},
                            headers=self.auth_headers,
                        )
                        self.assertEqual(resp.status_code, 503)
                        data = resp.json()
                        self.assertIn("Kimi (Moonshot AI) API key is unconfigured or invalid", data["detail"])

    def test_conversation_id_normalization(self):
        """Verify that string conversation IDs like 'conv-default' don't cause HTTP 422."""
        # Mock successful LLM call
        mock_res = {
            "content": "All clusters operating nominally under Kimi K3 analysis.",
            "model": "moonshotai/kimi-k3",
            "usage": {"total_tokens": 85},
            "suggested_actions": [],
            "cached": False,
        }
        with patch.object(llm_adapter, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_res
            resp = self.client.post(
                "/api/friday/chat",
                json={
                    "conversation_id": "conv-default",
                    "message": "Status report",
                    "context_hints": {"source_widget": "CrashRiskMeter", "liveCrashRisk": 4.2}
                },
                headers=self.auth_headers,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            # conversation_id must be a valid UUID
            uuid.UUID(data["conversation_id"])
            self.assertEqual(data["model_used"], "moonshotai/kimi-k3")
            self.assertEqual(data["response"], mock_res["content"])

    def test_kimi_tool_execution(self):
        """Verify that LLMAdapter correctly executes local tools when requested by Kimi."""
        biz_id = str(uuid.uuid4())
        
        # 1. Telemetry tool
        vitals = asyncio.run(llm_adapter.execute_tool("get_live_telemetry", {}, biz_id))
        self.assertIn("latency_p99_ms", vitals)
        self.assertIn("status", vitals)

        # 2. Anomalies tool
        anom_res = asyncio.run(llm_adapter.execute_tool("query_anomalies", {"limit": 3}, biz_id))
        self.assertIn("anomalies", anom_res)

        # 3. Action proposal tool
        action_args = {
            "action_type": "scale_service",
            "service": "checkout-v2",
            "params": {"replicas": 8},
            "rationale": "High traffic volume during flash sale"
        }
        action_res = asyncio.run(llm_adapter.execute_tool("propose_mitigation_action", action_args, biz_id))
        self.assertTrue(action_res["staged_action"])
        self.assertEqual(action_res["service"], "checkout-v2")
        self.assertEqual(action_res["params"]["replicas"], 8)

    def test_action_execution_endpoint(self):
        """Verify that confirmed mitigation actions trigger execution and return audit details."""
        resp = self.client.post(
            "/api/friday/actions/execute",
            json={
                "action_type": "scale_service",
                "service": "checkout-v2",
                "params": {"replicas": 8},
            },
            headers=self.auth_headers,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("checkout-v2", data["message"])
        self.assertIn("8 replicas", data["message"])
        self.assertTrue(data["details"]["audit_logged"])

    def test_health_probe_reports_kimi_status(self):
        """Verify that health check reports 'kimi_llm' service status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            resp = self.client.get("/api/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("kimi_llm", data["services"])
            self.assertNotIn("nvidia_nim", data["services"])

if __name__ == "__main__":
    unittest.main()
