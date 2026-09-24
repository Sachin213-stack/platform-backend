import unittest
import uuid
from datetime import timedelta
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token, create_refresh_token

class TestAuthRefresh(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.user_id = str(uuid.uuid4())
        self.business_id = str(uuid.uuid4())

    def test_successful_refresh(self):
        refresh_tok = create_refresh_token({
            "sub": self.user_id,
            "business_id": self.business_id,
            "role": "owner",
        })
        resp = self.client.post("/api/auth/refresh", json={"refresh_token": refresh_tok})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("access_token", data)
        self.assertIn("refresh_token", data)
        self.assertEqual(data["user_id"], self.user_id)
        self.assertEqual(data["business_id"], self.business_id)

    def test_access_token_rejected_as_refresh(self):
        # Access token cannot be used to refresh
        access_tok = create_access_token({
            "sub": self.user_id,
            "business_id": self.business_id,
            "role": "owner",
        })
        resp = self.client.post("/api/auth/refresh", json={"refresh_token": access_tok})
        self.assertEqual(resp.status_code, 401)
        self.assertIn("not a valid refresh token", resp.json()["detail"])

    def test_expired_refresh_token_rejected(self):
        # Expired refresh token
        expired_tok = create_refresh_token(
            {"sub": self.user_id, "business_id": self.business_id, "role": "owner"},
            expires_delta=timedelta(seconds=-10)
        )
        resp = self.client.post("/api/auth/refresh", json={"refresh_token": expired_tok})
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid or expired refresh token", resp.json()["detail"])

    def test_refresh_token_cannot_access_protected_endpoint(self):
        # Refresh token presented to /api/users/me should be rejected
        refresh_tok = create_refresh_token({
            "sub": self.user_id,
            "business_id": self.business_id,
            "role": "owner",
        })
        resp = self.client.get("/api/users/me", headers={"Authorization": f"Bearer {refresh_tok}"})
        self.assertEqual(resp.status_code, 401)
        self.assertIn("cannot be used for resource authentication", resp.json()["detail"])

if __name__ == "__main__":
    unittest.main()
