"""
Mid-run crash test for embedded MLWorker:
Confirms that when the worker task crashes mid-run (after starting successfully):
1. The watchdog logs a genuine 'terminated unexpectedly' ERROR.
2. The error is clearly distinguishable from the 'MLWorker failed to start:' startup failure log.
3. The API continues booting and serving /health and other endpoints without crashing.
"""

import asyncio
import logging
import httpx
from app.main import app
from app.core.logging import logger


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


async def test_midrun_crash():
    print("=" * 80)
    print("RUNNING MID-RUN CRASH TEST FOR MLWORKER LIFESPAN")
    print("=" * 80)

    capture_handler = LogCaptureHandler()
    capture_handler.setLevel(logging.ERROR)
    logger.addHandler(capture_handler)

    try:
        async with app.router.lifespan_context(app):
            print("Lifespan started successfully.")
            assert getattr(app.state, "worker_started_successfully", False) is True, \
                "Expected worker_started_successfully to be True"
            print("[PASS] Worker confirmed running (worker_started_successfully=True)")

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                # 1. Health check before crash
                res_health_pre = await client.get("/health")
                assert res_health_pre.status_code == 200
                print(f"[PASS] Pre-crash GET /health -> {res_health_pre.status_code}")

                # 2. Inject mid-run crash into the running worker task
                print("Injecting mid-run crash into running MLWorker task...")
                crash_msg = "Simulated mid-run worker crash: Redis connection dropped"
                app.state.inject_worker_crash(RuntimeError(crash_msg))

                # Wait briefly for the task and watchdog to process the exception
                await asyncio.sleep(0.3)

                # 3. Assert worker task is dead with the injected exception
                worker_task = app.state.worker_task
                assert worker_task.done(), "Worker task should be done after injected crash"
                exc = worker_task.exception()
                assert isinstance(exc, RuntimeError), f"Expected RuntimeError, got {type(exc)}"
                assert crash_msg in str(exc), f"Expected '{crash_msg}' in exception, got '{exc}'"
                print(f"[PASS] Worker task confirmed dead with injected exception: {exc}")

                # 4. Verify watchdog logged genuine 'terminated unexpectedly' ERROR
                error_messages = [r.getMessage() for r in capture_handler.records if r.levelno == logging.ERROR]
                unexpected_errors = [m for m in error_messages if "terminated unexpectedly with error" in m]
                startup_failure_errors = [m for m in error_messages if "MLWorker failed to start" in m]

                assert len(unexpected_errors) >= 1, f"Expected 'terminated unexpectedly' log, got: {error_messages}"
                assert len(startup_failure_errors) == 0, f"Did NOT expect startup failure log in mid-run crash, got: {startup_failure_errors}"
                print(f"[PASS] Watchdog emitted genuine 'terminated unexpectedly' ERROR: {unexpected_errors[0]}")
                print("[PASS] Confirmed mid-run crash log is completely distinguishable from startup-failure log")

                # 5. Confirm API continues serving /health and other endpoints without crashing
                res_health_post = await client.get("/health")
                assert res_health_post.status_code == 200
                print(f"[PASS] Post-crash GET /health -> {res_health_post.status_code} (API still healthy)")

                res_api_health = await client.get("/api/health")
                assert res_api_health.status_code == 200
                print(f"[PASS] Post-crash GET /api/health -> {res_api_health.status_code}")

                res_v1_health = await client.get("/api/v1/health")
                assert res_v1_health.status_code == 200
                print(f"[PASS] Post-crash GET /api/v1/health -> {res_v1_health.status_code}")

                # Demo token and dashboard metrics test while worker is dead
                res_demo = await client.post("/api/auth/demo")
                assert res_demo.status_code == 200
                token = res_demo.json()["access_token"]
                res_dash = await client.get("/api/v1/dashboard/metrics", headers={"Authorization": f"Bearer {token}"})
                assert res_dash.status_code == 200
                print(f"[PASS] Post-crash GET /api/v1/dashboard/metrics -> {res_dash.status_code}")

    finally:
        logger.removeHandler(capture_handler)

    print("\nALL MID-RUN CRASH CHECKS PASSED PERFECTLY!\n")


if __name__ == "__main__":
    asyncio.run(test_midrun_crash())
