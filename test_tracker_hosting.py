import asyncio
import httpx
from app.main import app

async def test_tracker():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test /static/tracker.js
        res_static = await client.get("/static/tracker.js")
        assert res_static.status_code == 200, f"Expected 200 on /static/tracker.js, got {res_static.status_code}"
        assert "javascript" in res_static.headers.get("content-type", "")
        assert "Access-Control-Allow-Origin" in res_static.headers
        assert res_static.headers["Access-Control-Allow-Origin"] == "*"
        print(f"[PASS] GET /static/tracker.js -> 200 (type={res_static.headers.get('content-type')})")

        # 2. Test /api/tracker.js alias
        res_api = await client.get("/api/tracker.js")
        assert res_api.status_code == 200, f"Expected 200 on /api/tracker.js, got {res_api.status_code}"
        print(f"[PASS] GET /api/tracker.js -> 200")

        # 3. Test /tracker.js root alias
        res_root = await client.get("/tracker.js")
        assert res_root.status_code == 200, f"Expected 200 on /tracker.js, got {res_root.status_code}"
        print(f"[PASS] GET /tracker.js -> 200")

        # 4. Verify script content
        content = res_static.text
        assert "data-business-id" in content, "Missing data-business-id in tracker.js"
        assert "data-api-key" in content, "Missing data-api-key in tracker.js"
        assert "X-API-Key" in content, "Missing X-API-Key header dispatch in tracker.js"
        assert "/api/ingestion/events" in content, "Missing ingestion route path in tracker.js"
        assert "localhost" not in content, "tracker.js must NOT contain hardcoded localhost URL!"
        assert "127.0.0.1" not in content, "tracker.js must NOT contain hardcoded 127.0.0.1 URL!"
        assert "http://" not in content and "https://" not in content, "tracker.js must derive origin dynamically from script src!"
        print(f"[PASS] tracker.js script content validated: no hardcoded hosts, dynamic origin extraction intact")

        print("\nALL PHASE 5 TRACKER HOSTING CHECKS PASSED 100%!")

if __name__ == "__main__":
    asyncio.run(test_tracker())
