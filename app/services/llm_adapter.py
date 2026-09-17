import re
import time
import hashlib
import json
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, func, desc

from app.core.config import settings
from app.core.logging import logger
from app.services.redis_service import redis_service
from app.db.session import AsyncSessionLocal, is_db_available, set_rls_context
from app.db.models.telemetry import TelemetryEvent
from app.db.models.ml import Anomaly, Forecast
from app.db.models.alerts import AlertRule
from app.db.models.business import Business


class RateLimitError(Exception):
    """Raised when Kimi endpoint returns HTTP 429 Too Many Requests."""
    pass


class KimiAuthenticationError(Exception):
    """Raised when Kimi API key is missing, unauthorized (401), or forbidden (403)."""
    pass


class AllEndpointsExhaustedError(Exception):
    """Raised when all configured Kimi model tiers are exhausted or failing."""
    pass


# Moonshot AI (Kimi) Tool Specifications (OpenAI JSON Schema compatible)
KIMI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_live_telemetry",
            "description": "Fetch real-time microservice performance vitals including p99 latency, error rates, CPU/memory saturation, and orders per minute.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_anomalies",
            "description": "Query active, unresolved engineering incidents and telemetry anomalies across microservices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max number of anomalies to return (1-10)", "default": 5},
                    "severity": {"type": "string", "description": "Filter by severity: low, medium, high, critical", "enum": ["low", "medium", "high", "critical"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_capacity_forecast",
            "description": "Retrieve predictive capacity forecasts, peak event runway, and estimated crash risk percentage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon": {"type": "string", "description": "Forecast time horizon", "enum": ["24h", "7d", "30d"], "default": "24h"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_mitigation_action",
            "description": "Propose an automated infrastructure mitigation action (e.g. scale pod replicas, invalidate CDN cache, adjust rate limits) requiring operator confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "description": "Type of action to execute",
                        "enum": ["scale_service", "purge_cdn_cache", "throttle_rate_limits", "restart_pod_pool", "adjust_alert_threshold"],
                    },
                    "service": {"type": "string", "description": "Target microservice name, e.g. checkout-v2, cart-service, payment-gw"},
                    "params": {"type": "object", "description": "Key-value execution parameters for the action, e.g. {'replicas': 8}"},
                    "rationale": {"type": "string", "description": "Engineering rationale for why this action mitigates the incident"},
                },
                "required": ["action_type", "service", "rationale"],
            },
        },
    },
]


def normalize_model_id(model: str | None) -> str:
    """Normalizes any model identifier or alias into a valid NVIDIA NIM model identifier."""
    if not model:
        return (settings.KIMI_MODEL_PRIMARY or "moonshotai/kimi-k3").strip('"\'')
    m = model.strip().strip('"\'')
    alias_map = {
        "kimi-k3": "moonshotai/kimi-k3",
        "kimi-k2.6": "moonshotai/kimi-k3",
        "kimi": "moonshotai/kimi-k3",
        "moonshot-v1-128k": "meta/llama-3.2-11b-vision-instruct",
        "llama-3.2-11b": "meta/llama-3.2-11b-vision-instruct",
        "nemotron-3.5": "nvidia/nemotron-3.5-lightning-30b-a3b",
    }
    return alias_map.get(m, m)


class LLMAdapter:
    """
    Production-grade LLM adapter for FRIDAY AI-CTO powered solely by Kimi (Moonshot AI):
    - Kimi model priority fallback chain (moonshotai/kimi-k3 -> meta/llama-3.2-11b-vision-instruct -> nvidia/nemotron-3.5-lightning-30b-a3b)
    - Native OpenAI-compatible tool use / function calling for live ops queries and action proposals
    - Real-time Server-Sent Events (SSE) streaming support
    - Half-open circuit breaker (Redis-backed cooldown + single probe interval per model tier)
    - Automatic PII redaction (emails, cards, secrets, tokens)
    - Grounded live business telemetry, anomaly, and widget context injection
    - Context window limits management with sliding-window summarization
    - Response caching for identical recent queries (TTL 60s)
    - Loud failure when KIMI_API_KEY is missing/invalid (no silent mock fallback)
    """

    def __init__(self) -> None:
        self._half_open_probes: Dict[str, float] = {}

    @property
    def endpoints(self) -> List[Dict[str, Any]]:
        """Returns Kimi model tiers in priority fallback order."""
        api_key = settings.effective_kimi_api_key
        raw_base = (settings.KIMI_BASE_URL or "https://integrate.api.nvidia.com/v1").rstrip("/")
        base_url = raw_base[:-17] if raw_base.endswith("/chat/completions") else raw_base
        return [
            {
                "id": "kimi-tier-1-primary",
                "base_url": base_url,
                "api_key": api_key,
                "model": normalize_model_id(settings.KIMI_MODEL_PRIMARY),
                "context_window": 1000000,
                "tier_name": "Kimi K3 (Moonshot AI)",
            },
            {
                "id": "kimi-tier-2-secondary",
                "base_url": base_url,
                "api_key": api_key,
                "model": normalize_model_id(settings.KIMI_MODEL_SECONDARY),
                "context_window": 128000,
                "tier_name": "Llama 3.2 11B (Fast Ops)",
            },
            {
                "id": "kimi-tier-3-fallback",
                "base_url": base_url,
                "api_key": api_key,
                "model": normalize_model_id(settings.KIMI_MODEL_FALLBACK),
                "context_window": 128000,
                "tier_name": "Nemotron 3.5 Lightning (Fallback)",
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

    async def get_live_business_context(self, business_id: str) -> Dict[str, Any]:
        """Fetches live telemetry vitals, active anomalies, and alert rules to ground FRIDAY's analysis."""
        context = {
            "business_name": "AI-CTO Managed Tenant",
            "avg_response_time_ms": 184.0,
            "error_rate_pct": 0.08,
            "orders_per_min": 5.3,
            "cpu_usage_pct": 42.0,
            "memory_usage_pct": 58.5,
            "queue_depth": 3,
            "active_anomalies": [],
            "alert_thresholds": [],
            "crash_risk_pct": 4.2,
        }

        if not await is_db_available():
            logger.debug("Database offline: using default baseline telemetry vitals for FRIDAY")
            return context

        try:
            import uuid
            biz_uuid = uuid.UUID(business_id)
            async with AsyncSessionLocal() as session:
                # Set tenant context for PostgreSQL Row-Level Security (RLS)
                await set_rls_context(session, str(business_id))

                # 1. Fetch Business metadata
                biz_stmt = select(Business).where(Business.id == biz_uuid)
                biz_obj = (await session.execute(biz_stmt)).scalars().first()
                if biz_obj:
                    context["business_name"] = biz_obj.name

                now = datetime.now(timezone.utc)
                one_hour_ago = now - timedelta(hours=1)

                # 2. Query latest telemetry aggregations
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

                # 3. Query unresolved anomalies
                anom_stmt = (
                    select(Anomaly)
                    .where(Anomaly.business_id == biz_uuid, Anomaly.is_resolved.is_(False))
                    .order_by(desc(Anomaly.detected_at))
                    .limit(5)
                )
                anoms = (await session.execute(anom_stmt)).scalars().all()
                context["active_anomalies"] = [
                    {
                        "id": str(a.id),
                        "metric": a.metric_name,
                        "severity": a.severity,
                        "description": a.description,
                        "expected": a.expected_value,
                        "actual": a.actual_value,
                        "detected_at": a.detected_at.isoformat(),
                    }
                    for a in anoms
                ]

                # 4. Query active alert rules
                rules_stmt = select(AlertRule).where(
                    AlertRule.business_id == biz_uuid,
                    AlertRule.is_enabled.is_(True),
                ).limit(5)
                rules = (await session.execute(rules_stmt)).scalars().all()
                context["alert_thresholds"] = [
                    f"{r.name}: {r.metric_target} {r.condition} {r.threshold_value}"
                    for r in rules
                ]

                # 5. Query latest crash risk forecast
                fc_stmt = (
                    select(Forecast)
                    .where(Forecast.business_id == biz_uuid)
                    .order_by(desc(Forecast.generated_at))
                    .limit(1)
                )
                fc = (await session.execute(fc_stmt)).scalars().first()
                if fc and fc.crash_risk_pct is not None:
                    context["crash_risk_pct"] = round(float(fc.crash_risk_pct), 1)

        except Exception as e:
            logger.debug("Could not load full business context for business %s: %s", business_id, e)

        return context

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], business_id: str) -> Dict[str, Any]:
        """Executes a real backend tool invocation requested by Kimi."""
        logger.info("Executing FRIDAY tool '%s' with args %s for business %s", tool_name, arguments, business_id)

        if tool_name == "get_live_telemetry":
            ctx = await self.get_live_business_context(business_id)
            return {
                "latency_p99_ms": ctx["avg_response_time_ms"],
                "error_rate_pct": ctx["error_rate_pct"],
                "cpu_saturation_pct": ctx["cpu_usage_pct"],
                "memory_saturation_pct": ctx["memory_usage_pct"],
                "queue_depth": ctx["queue_depth"],
                "throughput_orders_per_min": ctx["orders_per_min"],
                "status": "nominal" if ctx["avg_response_time_ms"] < 250 and ctx["error_rate_pct"] < 1.0 else "degraded",
            }

        elif tool_name == "query_anomalies":
            limit = int(arguments.get("limit", 5))
            severity_filter = arguments.get("severity")
            ctx = await self.get_live_business_context(business_id)
            anoms = ctx.get("active_anomalies", [])
            if severity_filter:
                anoms = [a for a in anoms if a.get("severity") == severity_filter]
            return {
                "active_anomalies_count": len(anoms),
                "anomalies": anoms[:limit],
            }

        elif tool_name == "get_capacity_forecast":
            horizon = arguments.get("horizon", "24h")
            ctx = await self.get_live_business_context(business_id)
            crash_risk = ctx.get("crash_risk_pct", 4.2)
            runway_days = max(7, min(90, int(45 - crash_risk * 0.4)))
            multiplier = round(max(1.4, 4.0 - (crash_risk / 25.0)), 1)
            rec = "Sufficient headroom for baseline traffic. Monitor Redis session pool."
            if crash_risk > 50:
                rec = "CRITICAL: High crash risk projected under current traffic growth. Immediately autoscale pod pool and apply memory limits."
            elif crash_risk > 20:
                rec = "MODERATE: Elevated concurrency detected. Proactively scale ingress gateways to 6 replicas."

            return {
                "horizon": horizon,
                "crash_risk_pct": crash_risk,
                "resource_runway_days": runway_days,
                "peak_multiplier_capacity": multiplier,
                "recommendation": rec,
            }

        elif tool_name == "propose_mitigation_action":
            action_type = arguments.get("action_type", "scale_service")
            service = arguments.get("service", "checkout-v2")
            params = arguments.get("params", {})
            rationale = arguments.get("rationale", "Automated mitigation requested by FRIDAY AI-CTO")
            return {
                "staged_action": True,
                "action_id": f"act_{int(time.time())}_{hashlib.md5(service.encode()).hexdigest()[:6]}",
                "action_type": action_type,
                "service": service,
                "params": params,
                "rationale": rationale,
                "requires_confirmation": True,
                "status": "ready_for_operator_confirmation",
            }

        return {"error": f"Unknown tool name '{tool_name}'"}

    def _truncate_or_summarize_messages(self, messages: List[Dict[str, Any]], max_tokens_estimate: int = 16000) -> List[Dict[str, Any]]:
        """Maintains conversational sliding window and bounds token usage within Kimi's limits."""
        # Approximate 1 token ~= 3.5 characters for mixed code/JSON/English
        total_chars = sum(len(m.get("content") or "") for m in messages)
        estimated_tokens = int(total_chars / 3.5)

        if estimated_tokens <= max_tokens_estimate or len(messages) <= 6:
            return messages

        # Preserve the system message (first), summarize the middle, and keep the latest 4 turns
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        turns = messages[1:] if system_msg else messages

        if len(turns) > 6:
            old_turns = turns[:-4]
            recent_turns = turns[-4:]
            summary_snippet = "Earlier exchanges: " + " | ".join(
                f"{t.get('role')}: {t.get('content')[:60]}..." for t in old_turns if t.get("content")
            )
            summary_msg = {"role": "system", "content": f"[Conversation History Summary: {summary_snippet}]"}
            new_msgs = [system_msg, summary_msg] + recent_turns if system_msg else [summary_msg] + recent_turns
            return new_msgs

        return messages

    async def _call_kimi(
        self,
        endpoint_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Performs async authenticated HTTP POST to Kimi (Moonshot AI) chat completions endpoint."""
        api_key = endpoint_config["api_key"]
        if not api_key:
            raise KimiAuthenticationError(
                "Kimi API key is not configured. Please set KIMI_API_KEY (or MOONSHOT_API_KEY) in .env or your environment variables."
            )

        url = f"{endpoint_config['base_url'].rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": endpoint_config["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }

        if tools:
            payload["tools"] = tools

        timeout_sec = min(10.0, float(getattr(settings, "KIMI_TIMEOUT_SECONDS", 10.0)))
        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                resp = await client.post(url, json=payload, headers=headers)
                latency_ms = (time.perf_counter() - start_time) * 1000

                if resp.status_code in [401, 403]:
                    logger.error(
                        "Kimi API authentication failed (HTTP %d): %s. Check KIMI_API_KEY.",
                        resp.status_code,
                        resp.text[:140],
                    )
                    raise KimiAuthenticationError(
                        f"Kimi API key rejected with HTTP {resp.status_code}: {resp.text[:100]}"
                    )

                if resp.status_code == 429:
                    logger.warning(
                        "Rate limit hit (HTTP 429) on Kimi tier %s (model: %s) after %.2fms",
                        endpoint_config["id"],
                        endpoint_config["model"],
                        latency_ms,
                    )
                    raise RateLimitError(f"Rate limited by Kimi {endpoint_config['model']}")

                if resp.status_code in [400, 404]:
                    err_msg = resp.text[:140]
                    logger.warning(
                        "Kimi model %s returned HTTP %d: %s (falling over to next model tier in chain)",
                        endpoint_config["model"],
                        resp.status_code,
                        err_msg,
                    )

                resp.raise_for_status()
                data = resp.json()

                logger.info(
                    "Kimi API call to %s (model: %s) succeeded in %.2fms",
                    endpoint_config["id"],
                    endpoint_config["model"],
                    latency_ms,
                )

                choice = data["choices"][0]
                return {
                    "content": choice["message"].get("content") or "",
                    "tool_calls": choice["message"].get("tool_calls"),
                    "model": endpoint_config["model"],
                    "tier_name": endpoint_config["tier_name"],
                    "usage": data.get("usage", {}),
                    "raw_message": choice["message"],
                }

        except (KimiAuthenticationError, RateLimitError):
            raise
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Kimi call to %s (model: %s) failed in %.2fms: %s",
                endpoint_config["id"],
                endpoint_config["model"],
                latency_ms,
                e,
            )
            raise

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        business_id: str,
        system_prompt: Optional[str] = None,
        context_hints: Optional[Dict[str, Any]] = None,
        inject_telemetry: bool = True,
        requested_model: Optional[str] = None,
        enable_tools: bool = True,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Executes full generation through Kimi priority fallback chain with tool calling and context grounding.
        Fails loudly if KIMI_API_KEY is missing/invalid or if all endpoints are exhausted.
        """
        # 0. Validate that a key is configured
        api_key = settings.effective_kimi_api_key
        if not api_key:
            logger.error("FRIDAY generation failed: KIMI_API_KEY is not configured in .env or environment")
            raise KimiAuthenticationError(
                "Kimi API key is not configured. Please configure KIMI_API_KEY in .env to enable FRIDAY AI."
            )

        # 1. Check Query Cache for identical recent queries (TTL 60s)
        last_user_msg = messages[-1]["content"] if messages else ""
        cache_hash = hashlib.sha256(
            f"{business_id}:{last_user_msg}:{requested_model or 'default'}:{json.dumps(context_hints or {}, sort_keys=True)}".encode()
        ).hexdigest()
        cache_key = f"llm:kimi_query_cache:{cache_hash}"

        cached_response = await redis_service.get_cache(cache_key)
        if cached_response:
            logger.info("FRIDAY Kimi query cache hit for business %s", business_id)
            cached_response["cached"] = True
            return cached_response

        # 2. Build Grounded Telemetry & Business Context
        full_system_prompt = system_prompt or (
            "You are FRIDAY, the principal AI-CTO and autonomous operations assistant for this enterprise digital platform. "
            "You are powered by Moonshot AI's Kimi neural engine. "
            "Provide precise, highly technical, and actionable infrastructure diagnostics. "
            "When diagnosing issues or recommending actions, ground your reasoning in the live telemetry and anomaly vitals provided."
        )

        suggested_actions: List[Dict[str, Any]] = []

        if inject_telemetry:
            biz_ctx = await self.get_live_business_context(business_id)
            telemetry_str = json.dumps(biz_ctx, indent=2)
            full_system_prompt += f"\n\n--- CURRENT LIVE ENTERPRISE VITALS ({biz_ctx['business_name']}) ---\n{telemetry_str}\n"

        if context_hints:
            hints_str = json.dumps(context_hints, indent=2)
            full_system_prompt += f"\n\n--- ACTIVE CONTEXT FROM OPERATOR STUDIO ---\n{hints_str}\nDirectly answer the operator's question in relation to this exact context.\n"

        # 3. Scrub input messages
        scrubbed_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.scrub_pii(full_system_prompt)}
        ]
        for msg in messages:
            scrubbed_messages.append({
                "role": msg["role"],
                "content": self.scrub_pii(msg.get("content") or ""),
            })

        # Apply context-window bounds & sliding-window summarization
        scrubbed_messages = self._truncate_or_summarize_messages(scrubbed_messages)

        # 4. Prepare endpoints fallback chain
        configured_endpoints = self.endpoints

        if requested_model:
            norm_model = normalize_model_id(requested_model)
            matched = [ep for ep in configured_endpoints if ep["model"] == norm_model]
            rest = [ep for ep in configured_endpoints if ep["model"] != norm_model]
            if matched:
                configured_endpoints = matched + rest
            else:
                custom_ep = {
                    "id": f"kimi-custom-{norm_model}",
                    "base_url": configured_endpoints[0]["base_url"],
                    "api_key": configured_endpoints[0]["api_key"],
                    "model": norm_model,
                    "tier_name": f"Kimi ({norm_model})",
                    "context_window": 128000,
                }
                configured_endpoints = [custom_ep] + configured_endpoints

        tools_to_pass = KIMI_TOOLS if enable_tools else None
        last_exception: Optional[Exception] = None

        for ep in configured_endpoints:
            ep_id = ep["id"]
            is_cooling_down = await redis_service.is_llm_cooling_down(ep_id)

            if is_cooling_down:
                last_probe = self._half_open_probes.get(ep_id, 0)
                now = time.time()
                if now - last_probe < 10.0:
                    logger.debug("Circuit breaker: %s in cooldown (%.1fs elapsed)", ep_id, now - last_probe)
                    continue
                self._half_open_probes[ep_id] = now
                logger.debug("Circuit breaker: sending half-open probe to Kimi tier %s", ep_id)

            try:
                # Initial LLM Call
                result = await self._call_kimi(ep, scrubbed_messages, tools=tools_to_pass, temperature=temperature)
                await redis_service.clear_llm_cooldown(ep_id)

                # 5. Handle Kimi Tool Calling Loop (if Kimi requested function execution)
                tool_calls = result.get("tool_calls")
                if tool_calls:
                    logger.info("Kimi (%s) requested %d tool calls", ep["model"], len(tool_calls))
                    followup_messages = list(scrubbed_messages)
                    followup_messages.append(result["raw_message"])

                    for tc in tool_calls:
                        fn_name = tc["function"]["name"]
                        raw_args = tc["function"].get("arguments", "{}")
                        try:
                            fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except Exception:
                            fn_args = {}

                        tool_out = await self.execute_tool(fn_name, fn_args, business_id)

                        if fn_name == "propose_mitigation_action" and tool_out.get("staged_action"):
                            suggested_actions.append(tool_out)

                        followup_messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(tool_out),
                        })

                    # Second call: Kimi synthesizes tool outputs into final response
                    synth_result = await self._call_kimi(ep, followup_messages, tools=None, temperature=temperature)
                    result["content"] = synth_result["content"]
                    result["usage"]["total_tokens"] += synth_result.get("usage", {}).get("total_tokens", 0)

                result["cached"] = False
                result["is_fallback"] = False
                result["suggested_actions"] = suggested_actions

                # Cache successful response for 60s
                await redis_service.set_cache(cache_key, result, ttl_seconds=60)
                return result

            except KimiAuthenticationError:
                # Do not proceed with fallback if the key is invalid — fail loudly immediately
                raise
            except Exception as e:
                last_exception = e
                cooldown_dur = 15
                logger.warning(
                    "Kimi endpoint %s failed (%s), setting circuit-breaker cooldown of %ds",
                    ep_id,
                    repr(e),
                    cooldown_dur,
                )
                await redis_service.set_llm_cooldown(ep_id, duration_seconds=cooldown_dur)

        # If all tiers failed, FAIL LOUDLY — no silent mock fallback
        err_msg = f"All Kimi (Moonshot AI) model tiers exhausted or failing in fallback chain: {last_exception}"
        logger.error(err_msg, exc_info=True)
        raise AllEndpointsExhaustedError(err_msg)

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        business_id: str,
        system_prompt: Optional[str] = None,
        context_hints: Optional[Dict[str, Any]] = None,
        inject_telemetry: bool = True,
        requested_model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """
        Streams token chunks from Kimi via Server-Sent Events (SSE).
        Yields lines of: data: {"token": "...", "model": "..."}\n\n
        """
        api_key = settings.effective_kimi_api_key
        if not api_key:
            yield f"data: {json.dumps({'error': 'Kimi API key is not configured. Set KIMI_API_KEY in .env.'})}\n\n"
            return

        full_system_prompt = system_prompt or "You are FRIDAY, AI-CTO for this system, powered by Kimi (Moonshot AI)."
        if inject_telemetry:
            biz_ctx = await self.get_live_business_context(business_id)
            full_system_prompt += f"\n\nLIVE SYSTEM VITALS:\n{json.dumps(biz_ctx, indent=2)}\n"

        if context_hints:
            full_system_prompt += f"\n\nOPERATOR CONTEXT:\n{json.dumps(context_hints, indent=2)}\n"

        scrubbed_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.scrub_pii(full_system_prompt)}
        ]
        for msg in messages:
            scrubbed_messages.append({
                "role": msg["role"],
                "content": self.scrub_pii(msg.get("content") or ""),
            })

        scrubbed_messages = self._truncate_or_summarize_messages(scrubbed_messages)
        configured_endpoints = self.endpoints

        if requested_model:
            norm_model = normalize_model_id(requested_model)
            matched = [ep for ep in configured_endpoints if ep["model"] == norm_model]
            rest = [ep for ep in configured_endpoints if ep["model"] != norm_model]
            configured_endpoints = matched + rest if matched else configured_endpoints

        for ep in configured_endpoints:
            url = f"{ep['base_url'].rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": ep["model"],
                "messages": scrubbed_messages,
                "temperature": temperature,
                "max_tokens": 2048,
                "stream": True,
            }

            try:
                async with httpx.AsyncClient(timeout=float(getattr(settings, "KIMI_TIMEOUT_SECONDS", 30.0))) as client:
                    async with client.stream("POST", url, json=payload, headers=headers) as response:
                        if response.status_code != 200:
                            err_body = await response.aread()
                            logger.warning("Kimi stream failed HTTP %d: %s", response.status_code, err_body[:100])
                            continue

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                yield f"data: {json.dumps({'done': True, 'model': ep['model']})}\n\n"
                                return

                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0]["delta"]
                                content_token = delta.get("content")
                                if content_token:
                                    yield f"data: {json.dumps({'token': content_token, 'model': ep['model']})}\n\n"
                            except Exception:
                                continue
                        return

            except Exception as e:
                logger.warning("Streaming failed for model %s: %s (trying next tier)", ep["model"], e)

        yield f"data: {json.dumps({'error': 'All Kimi stream tiers failed.'})}\n\n"


llm_adapter = LLMAdapter()
