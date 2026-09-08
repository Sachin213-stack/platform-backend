# AI-CTO Backend Structured Logging Guide

This document defines the structured logging standards, correlation tracing mechanisms, developer workflows, and zero-secret security policies for the AI-CTO modular monolith backend (`platform-backend`).

---

## 1. Core Architecture & Philosophy

The AI-CTO backend uses Python's standard library `logging` and `contextvars` to provide lightweight, enterprise-grade structured logging with zero heavy dependencies:

1. **Dual Outputs**:
   - **Console Handler (`sys.stdout`)**: In local development (`ENVIRONMENT=development`), outputs colored, human-readable log lines with correlation and tenant tags. In staging/production or when `LOG_FORMAT=json`, outputs JSON lines for platforms like Render and Datadog.
   - **Rotating File Handler (`logs/aicto.log`)**: Emits single-line, machine-parseable JSON objects. Rotates automatically at **10MB** per file and retains **5 rotated backups** (`logs/aicto.log.1`, etc.).

2. **Zero-Secret Logging Policy**:
   - **Never Log Sensitive Credentials**: Raw JWT tokens, user passwords, Fernet encryption keys, Redis auth strings, and API keys are strictly excluded.
   - **PII Scrubbing**: The FRIDAY LLM adapter scrubs emails, credit cards, and tokens prior to logging or dispatching prompts to external LLMs.

---

## 2. Distributed Tracing & Correlation IDs

Every inbound API request and background worker job runs with contextual tracing variables via `contextvars`:

- `correlation_id_ctx` (accessible via `get_correlation_id()` / `set_correlation_id(val)`):
  - Inbound HTTP requests: Extracted from `X-Correlation-ID` or `X-Request-ID` request headers. If absent, a UUID is automatically generated.
  - Workers & Jobs: Assigned a unique prefixed cycle ID (e.g., `ingest-a1b2c3d4`, `job-ml-anomaly-e5f6g7h8`).
  - Response Headers: Returned to clients in `X-Correlation-ID`, `X-Request-ID`, and `X-Process-Time-Ms`.
- `business_id_ctx` (accessible via `get_business_id()` / `set_business_id(val)`):
  - Populated immediately after JWT authentication or during tenant iteration in background jobs.
  - Enables instant multi-tenant filtering across all log records.

---

## 3. Log Levels & Standards

| Level | Purpose & Usage | Examples |
|---|---|---|
| **`DEBUG`** | Fine-grained diagnostic information for troubleshooting. | Duplicate idempotency key skip, circuit-breaker half-open probe, DB query fallback detail, telemetry event enqueue. |
| **`INFO`** | Normal operational milestones and performance metrics. | HTTP request completion (`GET /api/health -> 200 in 1.45ms`), batch write to Postgres with latency, user login/registration success, ML job start & finish. |
| **`WARNING`** | Handled errors, transient degradations, and potential security issues. | Failed login attempts (invalid password, deactivated account), invalid or expired JWT tokens, rate-limit hits (429) triggering LLM cooldown, Sentry init failure. |
| **`ERROR`** | Unhandled exceptions and operational failures requiring attention. | Database connection pool exhaustion, Redis Stream write failures, unhandled 500 exceptions, background worker loop crashes. |
| **`CRITICAL`** | Fatal system failures that cause service shutdown. | Database corruption or unrecoverable boot failures. |

---

## 4. Log Format Reference

### Structured JSON Format (`logs/aicto.log`)
Each line in `logs/aicto.log` is a standalone JSON object:

```json
{
  "timestamp": "2026-09-04T15:58:20.123456+00:00",
  "level": "INFO",
  "logger": "aicto",
  "message": "GET /api/dashboard/metrics -> 200 in 12.34ms",
  "correlation_id": "c7a8b9f1-3d2e-4b5a-9f8e-123456789abc",
  "request_id": "c7a8b9f1-3d2e-4b5a-9f8e-123456789abc",
  "business_id": "11111111-1111-1111-1111-111111111111",
  "module": "main",
  "line": 77,
  "exception": "Traceback (most recent call last):\n..."
}
```

### Local Console Format (`ENVIRONMENT=development`)
```text
[2026-09-04 21:28:43] [INFO   ] [aicto] [cid:c7a8b9f1 | biz:11111111] GET /api/dashboard/metrics -> 200 in 12.34ms
```

---

## 5. Developer Guide: How to Log

Always import the shared `logger` from `app.core.logging`:

```python
from app.core.logging import logger

# 1. Standard Info with string interpolation (never concatenate)
logger.info("Ingested %d events for tenant %s in %.2fms", count, business_id, duration_ms)

# 2. Warning with contextual details (never log passwords or tokens)
logger.warning("Auth failure on %s: Invalid credentials for email %s", request.url.path, email)

# 3. Error with stack trace (exc_info=True)
try:
    await do_work()
except Exception as e:
    logger.error("Failed to execute background task: %s", e, exc_info=True)

# 4. Debugging high-frequency events
logger.debug("Circuit breaker: endpoint %s in cooldown (%.1fs remaining)", ep_id, remaining)
```

---

## 6. Querying & Filtering Logs

### Filter by Business ID (Tenant Scoping)
Using `grep`:
```powershell
Get-Content logs/aicto.log | Select-String "11111111-1111-1111-1111-111111111111"
```
Using `jq`:
```bash
cat logs/aicto.log | jq 'select(.business_id == "11111111-1111-1111-1111-111111111111")'
```

### Trace a Single Request or Job by Correlation ID
```bash
cat logs/aicto.log | jq 'select(.correlation_id == "c7a8b9f1-3d2e-4b5a-9f8e-123456789abc")'
```

### Filter Errors and Exceptions
```bash
cat logs/aicto.log | jq 'select(.level == "ERROR" or .level == "CRITICAL") | {timestamp, message, exception}'
```

### Real-time Log Streaming
```powershell
Get-Content logs/aicto.log -Wait -Tail 20
```

---

## 7. Security Best Practices

1. **Never Log**:
   - `Authorization` header values (Bearer tokens).
   - Passwords, hashes, or salt strings.
   - Fernet decryption keys or API keys.
   - Customer credit card numbers or unredacted PII.
2. **Always Log**:
   - Relevant identifiers (UUIDs: `business_id`, `user_id`, `anomaly_id`, `conversation_id`).
   - Request paths and HTTP methods.
   - Durations in milliseconds for observability.
   - Failure reasons (`expired_token`, `invalid_password`, `deactivated_account`) without sensitive payloads.
