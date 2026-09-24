import httpx
import uuid
import sys
import json

BASE_URL = "http://127.0.0.1:8000"
results = []

def record(endpoint, method, status, expected, detail):
    passed = (status == expected)
    results.append({
        "endpoint": endpoint,
        "method": method,
        "status": status,
        "expected": expected,
        "passed": passed,
        "detail": detail
    })
    mark = "[PASS]" if passed else "[FAIL]"
    print(f"{mark} {method:<6} {endpoint:<32} -> HTTP {status} (expected {expected}) | {detail}")

def run_live_tests():
    print("=" * 80)
    print(f"LIVE HTTP ENDPOINT VALIDATION SUITE -> {BASE_URL}")
    print("=" * 80)

    client = httpx.Client(base_url=BASE_URL, timeout=20.0)

    # 1. Root
    r = client.get("/")
    data = r.json()
    record("/", "GET", r.status_code, 200, f"service: {data.get('service')} v{data.get('version')}")

    # 2. Docs (Swagger UI)
    r = client.get("/docs")
    record("/docs", "GET", r.status_code, 200, "Swagger UI documentation interactive shell")

    # 3. ReDoc
    r = client.get("/redoc")
    record("/redoc", "GET", r.status_code, 200, "ReDoc API specification documentation")

    # 4. OpenAPI JSON
    r = client.get("/api/openapi.json")
    spec = r.json()
    record("/api/openapi.json", "GET", r.status_code, 200, f"{len(spec.get('paths', {}))} routes documented in OpenAPI 3.1 schema")

    # 5. Health Check
    r = client.get("/api/health")
    h = r.json()
    services_summary = ", ".join([f"{k}: {v.get('status')}" for k, v in h.get("services", {}).items()])
    record("/api/health", "GET", r.status_code, 200, f"overall: {h.get('status')} ({services_summary})")

    # 6. Auth - Demo
    r = client.post("/api/auth/demo")
    demo_data = r.json()
    demo_token = demo_data.get("access_token")
    record("/api/auth/demo", "POST", r.status_code, 200, f"demo token generated for business {demo_data.get('business_id')[:8]}...")

    headers = {"Authorization": f"Bearer {demo_token}"}

    # 7. Auth - Me
    r = client.get("/api/auth/me", headers=headers)
    me_data = r.json()
    record("/api/auth/me", "GET", r.status_code, 200, f"user: {me_data.get('email')} ({me_data.get('role')})")

    # 8. Auth - Register new account
    reg_email = f"tester_{uuid.uuid4().hex[:6]}@example.com"
    r = client.post("/api/auth/register", json={
        "email": reg_email,
        "password": "TestPassword123!",
        "full_name": "Test Engineer",
        "business_name": "Apex QA Systems"
    })
    reg_data = r.json()
    record("/api/auth/register", "POST", r.status_code, 201, f"registered {reg_email}")

    # 9. Auth - Login
    r = client.post("/api/auth/login", json={
        "email": reg_email,
        "password": "TestPassword123!"
    })
    login_data = r.json()
    user_token = login_data.get("access_token")
    user_headers = {"Authorization": f"Bearer {user_token}"}
    record("/api/auth/login", "POST", r.status_code, 200, "authenticated with newly issued JWT")

    # 10. Dashboard Metrics
    r = client.get("/api/dashboard/metrics", headers=user_headers)
    metrics = r.json()
    kpis = metrics.get("kpis", {})
    record("/api/dashboard/metrics", "GET", r.status_code, 200, f"latency: {kpis.get('response_time_ms')}ms, error_rate: {kpis.get('error_rate_pct')}%, cache_hit: {metrics.get('cache_hit')}")

    # 11. Dashboard Metrics (2nd call for cache hit)
    r = client.get("/api/dashboard/metrics", headers=user_headers)
    m2 = r.json()
    record("/api/dashboard/metrics (cache)", "GET", r.status_code, 200, f"cache_hit: {m2.get('cache_hit')}")

    # 12. Telemetry Ingestion
    test_key = str(uuid.uuid4())
    r = client.post("/api/ingestion/events", json={
        "event_type": "order_completed",
        "response_time_ms": 142.5,
        "status_code": 200,
        "orders_count": 1,
        "revenue_amount": 79.99,
        "endpoint": "/checkout",
        "idempotency_key": test_key
    }, headers=user_headers)
    ingest_data = r.json()
    record("/api/ingestion/events", "POST", r.status_code, 202, f"event queued ({ingest_data.get('status')})")

    # 13. Telemetry Ingestion Duplicate Check
    r = client.post("/api/ingestion/events", json={
        "event_type": "order_completed",
        "idempotency_key": test_key
    }, headers=user_headers)
    dup_data = r.json()
    record("/api/ingestion/events (duplicate)", "POST", r.status_code, 202, f"duplicate ignored: {dup_data.get('duplicate')}")

    # 14. Friday AI Chat
    r = client.post("/api/friday/chat", json={
        "message": "Status check: What are our current latency and capacity vitals?"
    }, headers=user_headers)
    friday_data = r.json()
    resp_text = friday_data.get("response", "")[:75].replace("\n", " ")
    record("/api/friday/chat", "POST", r.status_code, 200, f"model: {friday_data.get('model_used')} | '{resp_text}...'")

    # 15. Auth - Logout (revocation)
    r = client.post("/api/auth/logout", headers=user_headers)
    record("/api/auth/logout", "POST", r.status_code, 200, "token revoked in Redis blacklist")

    # 16. Revoked token access
    r = client.get("/api/auth/me", headers=user_headers)
    record("/api/auth/me (revoked token)", "GET", r.status_code, 401, "denied access to blacklisted token")

    print("=" * 80)
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    print(f"RESULTS: {passed}/{total} endpoints passed successfully ({round(passed/total*100, 1)}%)")
    print("=" * 80)

    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    run_live_tests()
