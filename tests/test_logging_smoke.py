import os
import json
import uuid
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.security import create_access_token

def run_smoke_test():
    print("=== AI-CTO STRUCTURED LOGGING SMOKE TEST ===")

    # Ensure logging is initialized
    setup_logging()
    client = TestClient(app)

    log_file_path = settings.LOG_FILE_PATH or "logs/aicto.log"

    # 1. Health Probe Request
    custom_cid = f"test-cid-{uuid.uuid4().hex[:8]}"
    health_resp = client.get("/api/health", headers={"X-Correlation-ID": custom_cid})
    print(f"1. Health check: status={health_resp.status_code}")
    assert health_resp.status_code == 200, f"Expected 200, got {health_resp.status_code}"
    assert health_resp.headers.get("X-Correlation-ID") == custom_cid, "Custom correlation ID not returned"
    assert "X-Process-Time-Ms" in health_resp.headers, "Process time header missing"

    # 2. Invalid Token Request to /api/dashboard/metrics
    secret_token_canary = "SUPER_SECRET_CANARY_JWT_TOKEN_123456"
    auth_fail_resp = client.get(
        "/api/dashboard/metrics",
        headers={"Authorization": f"Bearer {secret_token_canary}"}
    )
    print(f"2. Auth failure check: status={auth_fail_resp.status_code}")
    assert auth_fail_resp.status_code == 401, f"Expected 401, got {auth_fail_resp.status_code}"

    # 3. FRIDAY Chat Call
    chat_resp = client.post(
        "/api/friday/chat",
        json={"message": "What is current server latency?", "mode": "chat"}
    )
    print(f"3. FRIDAY chat check: status={chat_resp.status_code}")
    assert chat_resp.status_code in [200, 503], f"Expected 200 or 503, got {chat_resp.status_code}"
    chat_data = chat_resp.json()
    if chat_resp.status_code == 200:
        assert "response" in chat_data, "No response field in chat output"
    else:
        assert "detail" in chat_data, "No error detail in chat output"

    # 4. Telemetry Ingestion with Idempotency Key (tested twice)
    idem_key = f"idem-test-{uuid.uuid4().hex}"
    event_payload = {
        "event_type": "request",
        "response_time_ms": 142.5,
        "status_code": 200,
        "idempotency_key": idem_key,
        "endpoint": "/api/checkout",
    }
    ingest_1 = client.post("/api/ingestion/events", json=event_payload)
    print(f"4a. Telemetry ingestion #1: status={ingest_1.status_code}, duplicate={ingest_1.json().get('duplicate')}")
    assert ingest_1.status_code == 202, f"Expected 202, got {ingest_1.status_code}"

    ingest_2 = client.post("/api/ingestion/events", json=event_payload)
    print(f"4b. Telemetry ingestion #2 (duplicate): status={ingest_2.status_code}, duplicate={ingest_2.json().get('duplicate')}")
    assert ingest_2.status_code == 202, f"Expected 202, got {ingest_2.status_code}"

    # 5. Background ML Job execution
    import asyncio
    from app.workers.ml_jobs import run_anomaly_detection_job
    print("5. Triggering ML anomaly detection job...")
    asyncio.run(run_anomaly_detection_job())
    print("   ML anomaly detection job executed.")

    # 6. Check log file content & JSON validity
    assert os.path.exists(log_file_path), f"Log file not found at {log_file_path}"
    
    with open(log_file_path, "r", encoding="utf-8") as f:
        log_lines = f.readlines()

    print(f"6. Inspecting {len(log_lines)} records in {log_file_path}...")
    assert len(log_lines) > 0, "No log lines recorded in log file"

    found_custom_cid = False
    found_auth_warning = False
    found_friday_info = False
    found_ml_job = False

    for line in log_lines:
        line_str = line.strip()
        if not line_str:
            continue
        try:
            entry = json.loads(line_str)
        except json.JSONDecodeError as err:
            raise AssertionError(f"Log line is not valid JSON: {line_str} (error: {err})")

        # Check required fields
        assert "timestamp" in entry, "Missing timestamp in log entry"
        assert "level" in entry, "Missing level in log entry"
        assert "logger" in entry, "Missing logger in log entry"
        assert "message" in entry, "Missing message in log entry"
        assert "correlation_id" in entry, "Missing correlation_id in log entry"
        assert "business_id" in entry, "Missing business_id in log entry"

        # Check canary secret is NEVER leaked
        assert secret_token_canary not in line_str, "CRITICAL: Secret token leaked in log file!"

        if entry.get("correlation_id") == custom_cid:
            found_custom_cid = True

        if entry.get("level") == "WARNING" and "Auth failure" in entry.get("message", ""):
            found_auth_warning = True

        if "FRIDAY chat completed" in entry.get("message", ""):
            found_friday_info = True

        if entry.get("correlation_id", "").startswith("job-ml-anomaly-"):
            found_ml_job = True

    assert found_custom_cid, f"Correlation ID {custom_cid} was not found in structured log file"
    assert found_auth_warning, "Expected WARNING level auth failure entry not found in structured log file"
    assert found_friday_info, "Expected FRIDAY chat completion entry not found in structured log file"
    assert found_ml_job, "Expected background ML job entry (job-ml-anomaly-*) not found in structured log file"

    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_smoke_test()
