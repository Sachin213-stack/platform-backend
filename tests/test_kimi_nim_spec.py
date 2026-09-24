import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.llm_adapter import llm_adapter, normalize_model_id, KimiAuthenticationError
from app.core.config import settings

def test_spec():
    print("Testing NVIDIA NIM moonshotai/kimi-k3 compliance...")

    # 1. Test model normalization strictly enforces moonshotai/kimi-k3
    assert normalize_model_id(None) == "moonshotai/kimi-k3"
    assert normalize_model_id("kimi-k3") == "moonshotai/kimi-k3"
    assert normalize_model_id("llama-3.2-11b") == "moonshotai/kimi-k3"
    assert normalize_model_id("arbitrary/model") == "moonshotai/kimi-k3"
    print("[PASS] normalize_model_id strictly enforces 'moonshotai/kimi-k3'")

    # 2. Test endpoints
    eps = llm_adapter.endpoints
    assert len(eps) == 1, f"Expected exactly 1 endpoint, got {len(eps)}"
    assert eps[0]["model"] == "moonshotai/kimi-k3"
    print("[PASS] llm_adapter.endpoints contains only 'moonshotai/kimi-k3'")

    # 3. Test multimodal PII scrubbing
    multimodal_content = [
        {"type": "text", "text": "Contact john.doe@example.com with key secret123456789012"},
        {"type": "image_url", "image_url": {"url": "https://assets.ngc.nvidia.com/test.jpg"}}
    ]
    scrubbed = llm_adapter.scrub_pii(multimodal_content)
    assert scrubbed[0]["text"] == "Contact [EMAIL_REDACTED] with key [REDACTED]"
    assert scrubbed[1]["image_url"]["url"] == "https://assets.ngc.nvidia.com/test.jpg"
    print("[PASS] PII scrubbing correctly handles multimodal content")

    # 4. Test _call_kimi payload format
    captured_kwargs = {}
    async def mock_post(self, url, **kwargs):
        captured_kwargs["url"] = url
        captured_kwargs["json"] = kwargs.get("json")
        captured_kwargs["headers"] = kwargs.get("headers")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Kimi K3 response", "tool_calls": None}}],
            "usage": {"total_tokens": 100},
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", new=mock_post):
        res = asyncio.run(llm_adapter._call_kimi(
            eps[0],
            messages=[{"role": "user", "content": "Analyze cluster"}],
            temperature=1,
        ))
        payload = captured_kwargs["json"]
        headers = captured_kwargs["headers"]

        assert payload["model"] == "moonshotai/kimi-k3", f"Wrong model: {payload.get('model')}"
        assert payload["max_tokens"] == 16384, f"Wrong max_tokens: {payload.get('max_tokens')}"
        assert payload["seed"] == 0, f"Wrong seed: {payload.get('seed')}"
        assert payload["temperature"] == 1, f"Wrong temperature: {payload.get('temperature')}"
        assert payload["reasoning_effort"] == "max", f"Wrong reasoning_effort: {payload.get('reasoning_effort')}"
        assert headers["Accept"] == "application/json"
        assert "Bearer " in headers["Authorization"]
        print("[PASS] _call_kimi sends exact NVIDIA NIM payload (model='moonshotai/kimi-k3', max_tokens=16384, seed=0, temp=1, reasoning_effort='max')")

    # 5. Test generate_stream payload format
    async def run_stream():
        stream_kwargs = {}
        class MockStreamResponse:
            status_code = 200
            async def aiter_lines(self):
                yield 'data: {"choices": [{"delta": {"content": "Hello from Kimi"}}]}'
                yield 'data: [DONE]'

        class MockAsyncClient:
            def __init__(self, **kwargs):
                stream_kwargs.update(kwargs)
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def stream(self, method, url, json=None, headers=None):
                stream_kwargs["url"] = url
                stream_kwargs["json"] = json
                stream_kwargs["headers"] = headers
                class StreamCtx:
                    async def __aenter__(self_ctx):
                        return MockStreamResponse()
                    async def __aexit__(self_ctx, *args):
                        pass
                return StreamCtx()

        with patch("httpx.AsyncClient", new=MockAsyncClient):
            tokens = []
            async for chunk in llm_adapter.generate_stream(
                messages=[{"role": "user", "content": "Stream test"}],
                business_id="00000000-0000-0000-0000-000000000001",
                inject_telemetry=False,
            ):
                tokens.append(chunk)

            s_payload = stream_kwargs["json"]
            s_headers = stream_kwargs["headers"]
            assert s_payload["model"] == "moonshotai/kimi-k3"
            assert s_payload["max_tokens"] == 16384
            assert s_payload["seed"] == 0
            assert s_payload["temperature"] == 1
            assert s_payload["reasoning_effort"] == "max"
            assert s_payload["stream"] is True
            assert s_headers["Accept"] == "text/event-stream"
            assert len(tokens) >= 1
            print("[PASS] generate_stream sends exact NVIDIA NIM SSE payload (Accept: text/event-stream, stream=True)")

    asyncio.run(run_stream())
    print("\nALL NVIDIA NIM MOONSHOTAI/KIMI-K3 SPECIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_spec()
