import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.llm_adapter import llm_adapter
from app.api.schemas.friday import FridayChatRequest

def test_reasoning_effort_mapping():
    print("Testing reasoning_effort payload mapping...")
    eps = llm_adapter.endpoints
    captured_kwargs = {}

    async def mock_post(self, url, **kwargs):
        captured_kwargs["json"] = kwargs.get("json")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "OK", "tool_calls": None}}],
            "usage": {"total_tokens": 10},
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", new=mock_post):
        # 1. Test "low"
        asyncio.run(llm_adapter._call_kimi(eps[0], [{"role": "user", "content": "test"}], reasoning_effort="low"))
        assert captured_kwargs["json"]["max_tokens"] == 2048
        assert captured_kwargs["json"]["reasoning_effort"] == "low"
        print("[PASS] reasoning_effort='low' correctly sets max_tokens=2048, reasoning_effort='low'")

        # 2. Test "medium"
        asyncio.run(llm_adapter._call_kimi(eps[0], [{"role": "user", "content": "test"}], reasoning_effort="medium"))
        assert captured_kwargs["json"]["max_tokens"] == 4096
        assert captured_kwargs["json"]["reasoning_effort"] == "medium"
        print("[PASS] reasoning_effort='medium' correctly sets max_tokens=4096, reasoning_effort='medium'")

        # 3. Test "max"
        asyncio.run(llm_adapter._call_kimi(eps[0], [{"role": "user", "content": "test"}], reasoning_effort="max"))
        assert captured_kwargs["json"]["max_tokens"] == 16384
        assert captured_kwargs["json"]["reasoning_effort"] == "max"
        print("[PASS] reasoning_effort='max' correctly sets max_tokens=16384, reasoning_effort='max'")


def test_approach_a_failover_and_grounded_fallback():
    print("Testing Approach A resilient failover & grounded telemetry...")

    # 1. Simulate Kimi failure -> Fast Failover success
    async def run_failover_test():
        with patch.object(llm_adapter, "_call_kimi", side_effect=Exception("Upstream Kimi K3 timeout")):
            with patch.object(llm_adapter, "_call_fallback_model", new=AsyncMock(return_value={
                "content": "Diagnostics normal via failover.",
                "tool_calls": None,
                "model": "meta/llama-3.2-11b-vision-instruct (Fast Failover)",
                "tier_name": "Fast Resilient Failover",
                "usage": {"total_tokens": 50},
                "cached": False,
                "is_fallback": True,
                "suggested_actions": [],
            })):
                res = await llm_adapter.generate(
                    messages=[{"role": "user", "content": "Status check"}],
                    business_id="00000000-0000-0000-0000-000000000001",
                    reasoning_effort="medium",
                )
                assert "Fast Failover" in res["model"]
                assert res["is_fallback"] is True
                assert res["content"] == "Diagnostics normal via failover."
                print("[PASS] Upstream Kimi failure successfully triggers Fast Failover to meta/llama-3.2")

    asyncio.run(run_failover_test())

    # 2. Simulate complete failure (both Kimi & fallback down) -> Grounded Telemetry Fallback
    async def run_grounded_test():
        with patch.object(llm_adapter, "_call_kimi", side_effect=Exception("Kimi cluster unreachable")):
            with patch.object(llm_adapter, "_call_fallback_model", side_effect=Exception("Fallback unreachable")):
                res = await llm_adapter.generate(
                    messages=[{"role": "user", "content": "Status check"}],
                    business_id="00000000-0000-0000-0000-000000000001",
                    reasoning_effort="low",
                )
                assert res["model"] == "grounded-telemetry-engine"
                assert "FRIDAY Operations Status" in res["content"]
                assert "P99 Latency" in res["content"]
                print("[PASS] Total upstream outage cleanly triggers Grounded Telemetry Fallback (0 errors to user)")

    asyncio.run(run_grounded_test())


def test_schema_validation():
    print("Testing FridayChatRequest schema validation for reasoning_effort...")
    req_default = FridayChatRequest(message="Hello")
    assert req_default.reasoning_effort == "medium"

    req_low = FridayChatRequest(message="Hello", reasoning_effort="low")
    assert req_low.reasoning_effort == "low"

    req_max = FridayChatRequest(message="Hello", reasoning_effort="max")
    assert req_max.reasoning_effort == "max"
    print("[PASS] FridayChatRequest schema validates reasoning_effort (default='medium', supports 'low', 'max')")


if __name__ == "__main__":
    test_reasoning_effort_mapping()
    test_approach_a_failover_and_grounded_fallback()
    test_schema_validation()
    print("\nALL REASONING EFFORT & APPROACH A FAILOVER TESTS PASSED!")
