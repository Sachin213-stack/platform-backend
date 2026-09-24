import unittest
from unittest.mock import patch, AsyncMock
import uuid
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.friday import router as friday_router, FRIDAY_VOICE_SYSTEM_PROMPT
from app.api.dependencies.auth import get_current_user_and_business
from app.services.voice_service import clean_text_for_speech, voice_service
from app.services.llm_adapter import llm_adapter


class MockUser:
    id = uuid.uuid4()
    business_id = uuid.uuid4()
    email = "operator@aicto.io"
    role = "owner"


class TestVoiceSynthesis(unittest.TestCase):
    def setUp(self):
        self.test_app = FastAPI()
        self.test_app.include_router(friday_router)
        self.test_app.dependency_overrides[get_current_user_and_business] = lambda: MockUser()
        self.client = TestClient(self.test_app)

    def test_clean_text_for_speech(self):
        raw = "Alert: **p99 latency** is `280ms` (jumped 15%). See [Dashboard](https://app.io).\n| Svc | Latency |\n|---|---|\n| checkout | 180ms |\n```python\nprint(1)\n```"
        cleaned = clean_text_for_speech(raw)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("|", cleaned)
        self.assertNotIn("[", cleaned)
        self.assertNotIn("]", cleaned)
        self.assertIn("P 99 latency", cleaned)
        self.assertIn("milliseconds", cleaned)
        self.assertIn("percent", cleaned)
        self.assertIn("transcript", cleaned)

    def test_voice_mapping_resolution(self):
        self.assertEqual(voice_service.resolve_voice("friday-core-female"), "en-US-AriaNeural")
        self.assertEqual(voice_service.resolve_voice("friday-nova-neutral"), "en-GB-SoniaNeural")
        self.assertEqual(voice_service.resolve_voice("friday-echo-male"), "en-US-GuyNeural")
        self.assertEqual(voice_service.resolve_voice("friday-solis-female"), "hi-IN-SwaraNeural")
        self.assertEqual(voice_service.resolve_voice("unknown-voice"), "en-US-AriaNeural")

    def test_voice_synthesize_endpoint_mock(self):
        """Verify POST /friday/voice/synthesize returns audio/mpeg stream."""
        async def mock_stream(text, voice_id=None, rate=1.0, pitch="+0Hz"):
            yield b"\xff\xfb\x90\x64"  # MP3 frame header
            yield b"\x00" * 32

        with patch.object(voice_service, "stream_speech", side_effect=mock_stream):
            resp = self.client.post(
                "/friday/voice/synthesize",
                json={
                    "text": "All systems operating within acceptable latency thresholds.",
                    "voice": "en-US-AriaNeural",
                    "rate": 1.0,
                },
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers.get("content-type"), "audio/mpeg")
            self.assertGreater(len(resp.content), 0)

    def test_voice_mode_uses_voice_prompt(self):
        """Verify chat_with_friday uses FRIDAY_VOICE_SYSTEM_PROMPT when mode == 'voice'."""
        mock_res = {
            "content": "Checkout latency is 180 milliseconds. Everything looks healthy.",
            "model": "moonshotai/kimi-k3",
            "usage": {"total_tokens": 40},
            "suggested_actions": [],
            "cached": False,
        }
        with patch("app.api.routes.friday.redis_service.get_session_memory", new_callable=AsyncMock) as mock_redis_get, \
             patch("app.api.routes.friday.redis_service.save_session_memory", new_callable=AsyncMock) as mock_redis_save, \
             patch("app.api.routes.friday.is_db_available", new_callable=AsyncMock) as mock_db, \
             patch.object(llm_adapter, "generate", new_callable=AsyncMock) as mock_gen:
            mock_redis_get.return_value = []
            mock_db.return_value = False
            mock_gen.return_value = mock_res

            resp = self.client.post(
                "/friday/chat",
                json={
                    "message": "Give me a quick cluster update.",
                    "mode": "voice",
                },
            )
            self.assertEqual(resp.status_code, 200)
            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args.kwargs
            self.assertEqual(call_kwargs.get("system_prompt"), FRIDAY_VOICE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
