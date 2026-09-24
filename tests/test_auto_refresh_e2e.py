import unittest
import uuid
from datetime import timedelta
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token, create_refresh_token
from app.services.llm_adapter import llm_adapter

class TestAutoRefreshEndToEnd(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.user_id = str(uuid.uuid4())
        self.business_id = str(uuid.uuid4())

    def test_e2e_token_expiry_refresh_and_chat_continuity(self):
        # 1. User starts with an access token that is already expired (e.g., 35 minutes old)
        expired_access_token = create_access_token(
            {
                "sub": self.user_id,
                "business_id": self.business_id,
                "role": "owner",
                "email": "lead.cto@apex.io",
                "name": "Alex Vance",
            },
            expires_delta=timedelta(seconds=-1)
        )
        
        # Valid 7-day refresh token
        valid_refresh_token = create_refresh_token({
            "sub": self.user_id,
            "business_id": self.business_id,
            "role": "owner",
            "email": "lead.cto@apex.io",
            "name": "Alex Vance",
        })

        # 2. Direct call to /api/users/me with expired token fails with 401
        resp_expired = self.client.get(
            "/api/users/me",
            headers={"Authorization": f"Bearer {expired_access_token}"}
        )
        self.assertEqual(resp_expired.status_code, 401)
        self.assertIn("Invalid or expired authentication token", resp_expired.json()["detail"])

        # 3. Direct call to /api/friday/chat with expired token also fails with 401
        resp_friday_expired = self.client.post(
            "/api/friday/chat",
            json={"message": "Report cluster latency"},
            headers={"Authorization": f"Bearer {expired_access_token}"}
        )
        self.assertEqual(resp_friday_expired.status_code, 401)
        self.assertIn("Invalid or expired authentication token", resp_friday_expired.json()["detail"])

        # 4. Auto-refresh call to /api/auth/refresh with valid refresh token succeeds
        resp_refresh = self.client.post(
            "/api/auth/refresh",
            json={"refresh_token": valid_refresh_token}
        )
        self.assertEqual(resp_refresh.status_code, 200)
        refresh_data = resp_refresh.json()
        new_access_token = refresh_data["access_token"]
        new_refresh_token = refresh_data["refresh_token"]
        self.assertIsNotNone(new_access_token)
        self.assertNotEqual(new_access_token, expired_access_token)

        # 5. Retry /api/users/me with fresh access token succeeds with 200
        new_headers = {"Authorization": f"Bearer {new_access_token}"}
        resp_me = self.client.get("/api/users/me", headers=new_headers)
        self.assertEqual(resp_me.status_code, 200)
        me_data = resp_me.json()
        self.assertEqual(me_data["email"], "lead.cto@apex.io")

        # 6. FRIDAY chat with fresh access token succeeds with 200
        mock_llm_res = {
            "content": "All edge cluster vitals nominal following session refresh.",
            "model": "kimi-k3",
            "usage": {"total_tokens": 42},
            "suggested_actions": [],
            "cached": False,
        }
        with patch.object(llm_adapter, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_llm_res
            resp_chat = self.client.post(
                "/api/friday/chat",
                json={"message": "Report cluster latency"},
                headers=new_headers
            )
            self.assertEqual(resp_chat.status_code, 200)
            self.assertEqual(resp_chat.json()["response"], mock_llm_res["content"])

        # 7. When refresh token itself expires, refresh fails with 401
        expired_refresh_token = create_refresh_token(
            {"sub": self.user_id, "business_id": self.business_id, "role": "owner"},
            expires_delta=timedelta(seconds=-1)
        )
        resp_failed_refresh = self.client.post(
            "/api/auth/refresh",
            json={"refresh_token": expired_refresh_token}
        )
        self.assertEqual(resp_failed_refresh.status_code, 401)
        self.assertIn("Invalid or expired refresh token", resp_failed_refresh.json()["detail"])

if __name__ == "__main__":
    unittest.main()
