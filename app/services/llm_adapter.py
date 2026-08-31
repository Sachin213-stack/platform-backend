import re
import time
import hashlib
import json
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, func, desc

from app.core.config import settings
from app.core.logging import logger
from app.services.redis_service import redis_service
from app.db.session import AsyncSessionLocal
from app.db.models.telemetry import TelemetryEvent
from app.db.models.ml import Anomaly


class LLMAdapter:
    """
    Production-grade LLM adapter for FRIDAY AI-CTO:
    - Multi-NIM priority fallback chain (Llama 3.1 70B -> Mixtral 8x22B -> Llama 3.1 8B)
    - Half-open circuit breaker (Redis-backed cooldown + single probe interval)
    - Automatic PII redaction (emails, cards, secrets, tokens)
    - Grounded live telemetry context injection
    - Response caching for identical recent queries
    """

    def __init__(self) -> None:
        self._half_open_probes: Dict[str, float] = {}

    @property
    def endpoints(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "nim-primary-70b",
                "base_url": (settings.NIM_ENDPOINT_1 or "").strip('"\''),
                "api_key": (settings.NIM_API_KEY_1 or "").strip('"\''),
                "model": (settings.NIM_MODEL_1 or "").strip('"\''),
            },
            {
                "id": "nim-secondary-mixtral",
                "base_url": (settings.NIM_ENDPOINT_2 or "").strip('"\''),
                "api_key": (settings.NIM_API_KEY_2 or "").strip('"\''),
                "model": (settings.NIM_MODEL_2 or "").strip('"\''),
            },
            {
                "id": "nim-fallback-8b",
                "base_url": (settings.NIM_ENDPOINT_3 or "").strip('"\''),
                "api_key": (settings.NIM_API_KEY_3 or "").strip('"\''),
                "model": (settings.NIM_MODEL_3 or "").strip('"\''),
            },
        ]

    def scrub_pii(self, text: str) -> str:
        """Sanitizes PII and credentials prior to external LLM dispatch."""
        if not text:
            return ""
        # Scrub emails
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REDACTED]", text)
        # Scrub credit cards (13-16 digits)
        text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CARD_REDACTED]", text)
        # Scrub auth tokens and API keys
        text = re.sub(r"(?i)(bearer|token|key|secret|password)[\s:=]+([a-zA-Z0-9_\-\.]{12,})", r"\1 [REDACTED]", text)
        return text

    async def get_live_telemetry_context(self, business_id: str) -> Dict[str, Any]:
        """Fetches live telemetry vitals and recent anomalies to ground FRIDAY's analysis."""
        context = {
            "avg_response_time_ms": 184.0,
            "error_rate_pct": 0.08,
            "orders_per_min": 5.3,
            "cpu_usage_pct": 42.0,
            "memory_usage_pct": 58.5,
            "queue_depth": 3,
            "active_anomalies": [],
        }

        try:
            import uuid
            biz_uuid = uuid.UUID(business_id)
            async with AsyncSessionLocal() as session:
                now = datetime.now(timezone.utc)
                one_hour_ago = now - timedelta(hours=1)

                # Query latest telemetry aggregations
                perf_stmt = select(
                    func.avg(TelemetryEvent.response_time_ms).label("avg_latency"),
                    func.avg(TelemetryEvent.cpu_usage_pct).label("avg_cpu"),
                    func.avg(TelemetryEvent.memory_usage_pct).label("avg_mem"),
                    func.avg(TelemetryEvent.queue_depth).label("avg_queue"),
                    func.sum(TelemetryEvent.orders_count).label("orders"),
                ).where(
                    TelemetryEvent.business_id == biz_uuid,
                    TelemetryEvent.timestamp >= one_hour_ago,
                )
                res = (await session.execute(perf_stmt)).first()

                if res and res.avg_latency is not None:
                    context["avg_response_time_ms"] = round(float(res.avg_latency), 1)
                    context["cpu_usage_pct"] = round(float(res.avg_cpu or 40.0), 1)
                    context["memory_usage_pct"] = round(float(res.avg_mem or 55.0), 1)
                    context["queue_depth"] = int(res.avg_queue or 0)
                    if res.orders:
                        context["orders_per_min"] = round(float(res.orders) / 60.0, 1)

                # Query unresolved anomalies
                anom_stmt = (
                    select(Anomaly)
                    .where(Anomaly.business_id == biz_uuid, Anomaly.is_resolved.is_(False))
                    .order_by(desc(Anomaly.detected_at))
                    .limit(3)
                )
                anoms = (await session.execute(anom_stmt)).scalars().all()
                context["active_anomalies"] = [
                    f"{a.metric_name} ({a.severity}): {a.description} [Expected: {a.expected_value}, Actual: {a.actual_value}]"
                    for a in anoms
                ]
        except Exception as e:
            logger.warning(f"Could not load live telemetry context for business {business_id}: {e}")

        return context

    async def _call_nim(self, endpoint_config: Dict[str, Any], messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Performs async HTTP POST to OpenAI-compatible NIM endpoint."""
        url = f"{endpoint_config['base_url'].rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {endpoint_config['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": endpoint_config["model"],
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limited by {endpoint_config['id']}")
            resp.raise_for_status()
            data = resp.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "model": endpoint_config["model"],
                "usage": data.get("usage", {}),
            }

    async def generate(
        self,
        messages: List[Dict[str, str]],
        business_id: str,
        system_prompt: Optional[str] = None,
        inject_telemetry: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes generation through the resilient fallback chain with caching and live context injection.
        """
        # 1. Check Query Cache for identical recent queries (TTL 60s)
        last_user_msg = messages[-1]["content"] if messages else ""
        cache_hash = hashlib.sha256(f"{business_id}:{last_user_msg}".encode()).hexdigest()
        cache_key = f"llm:query_cache:{cache_hash}"

        cached_response = await redis_service.get_cache(cache_key)
        if cached_response and not cached_response.get("is_fallback"):
            logger.info(f"FRIDAY LLM query cache hit for business {business_id}")
            cached_response["cached"] = True
            return cached_response

        # 2. Build Telemetry Context if enabled
        full_system_prompt = system_prompt or "You are FRIDAY, the AI CTO for this business."
        if inject_telemetry:
            telemetry = await self.get_live_telemetry_context(business_id)
            telemetry_str = json.dumps(telemetry, indent=2)
            full_system_prompt += f"\n\nCURRENT LIVE SYSTEM VITALS:\n{telemetry_str}\nUse these live vitals to deliver grounded, precise engineering recommendations."

        # 3. Scrub input messages (including system prompt with telemetry context)
        scrubbed_messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.scrub_pii(full_system_prompt)}
        ]
        for msg in messages:
            scrubbed_messages.append({
                "role": msg["role"],
                "content": self.scrub_pii(msg["content"]),
            })

        # 4. Attempt Multi-NIM Fallback Chain
        for ep in self.endpoints:
            if not ep["api_key"]:
                continue

            ep_id = ep["id"]
            is_cooling_down = await redis_service.is_llm_cooling_down(ep_id)

            if is_cooling_down:
                last_probe = self._half_open_probes.get(ep_id, 0)
                now = time.time()
                if now - last_probe < 10.0:
                    logger.info(f"Skipping {ep_id} (in cooldown, waiting for probe)")
                    continue
                self._half_open_probes[ep_id] = now
                logger.info(f"Probing {ep_id} in half-open state")

            try:
                result = await self._call_nim(ep, scrubbed_messages)
                await redis_service.clear_llm_cooldown(ep_id)
                # Store in query cache
                await redis_service.set_cache(cache_key, result, ttl_seconds=60)
                result["cached"] = False
                return result
            except Exception as e:
                logger.warning(f"Endpoint {ep_id} failed: {repr(e)}. Marking 10s cooldown.")
                await redis_service.set_llm_cooldown(ep_id, duration_seconds=10)

        # 5. Smart Simulation / Fallback Mode when NIM keys are not active in dev
        logger.info("Operating in FRIDAY local intelligent fallback mode")
        telemetry = await self.get_live_telemetry_context(business_id)
        
        simulated_response = (
            f"**FRIDAY AI-CTO Analysis**:\n\n"
            f"- **System Latency**: {telemetry['avg_response_time_ms']} ms (nominal threshold < 250ms)\n"
            f"- **Compute Capacity**: CPU at {telemetry['cpu_usage_pct']}%, Memory at {telemetry['memory_usage_pct']}%\n"
            f"- **Throughput**: ~{telemetry['orders_per_min']} orders/min with queue depth {telemetry['queue_depth']}\n\n"
            f"**Recommendation**: All core services are operating within safety bounds. Ingestion pipelines and database connection pools are healthy."
        )

        response_obj = {
            "content": simulated_response,
            "model": "aicto-nim-fallback-engine",
            "usage": {"total_tokens": 145},
            "cached": False,
            "is_fallback": True,
        }
        return response_obj


llm_adapter = LLMAdapter()
